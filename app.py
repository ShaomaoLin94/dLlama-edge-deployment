from __future__ import annotations

from collections import deque
import os
import threading
import time

from flask import Flask, abort, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    AudioMessage,
    FileMessage,
    ImageMessage,
    LocationMessage,
    MessageEvent,
    StickerMessage,
    TextMessage,
    TextSendMessage,
    VideoMessage,
)
from requests.exceptions import RequestException

from call import InferenceResult, InferenceTask
from cluster import ClusterManager, load_worker_specs


LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise RuntimeError(
        "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set as environment variables"
    )

INFERENCE_STEPS = int(os.environ.get("DLLAMA_STEPS", "30"))
MAX_RECOVERY_RETRIES = int(os.environ.get("DLLAMA_MAX_RECOVERY_RETRIES", "3"))
RECOVERY_SETTLE_SEC = float(os.environ.get("DLLAMA_RECOVERY_SETTLE_SEC", "0.5"))


app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

cluster = ClusterManager(load_worker_specs())
cluster.start()

# user session / request queue
user_modes: dict[str, str] = {}
user_waiting: deque[tuple[str, str]] = deque()
waiting_lock = threading.Lock()
running = False

# LINE can retry a webhook when acknowledgement is delayed.
processed_lock = threading.Lock()
PROCESSED: dict[str, float] = {}
TTL_SECONDS = 600


def safe_reply(reply_token, messages, timeout=10):
    try:
        line_bot_api.reply_message(reply_token, messages, timeout=timeout)
        return True
    except LineBotApiError as exc:
        if getattr(exc, "status_code", None) == 401:
            print(
                "[line] reply failed: invalid LINE channel access token (HTTP 401). "
                "Check LINE_CHANNEL_ACCESS_TOKEN."
            )
        else:
            print(f"[line] reply failed: {exc}")
        return False
    except RequestException as exc:
        print(f"[line] network error while replying: {exc}")
        return False
    except Exception as exc:
        print(f"[line] unexpected reply error: {exc}")
        return False


def safe_push(to, messages, timeout=10):
    try:
        line_bot_api.push_message(to, messages, timeout=timeout)
        return True
    except LineBotApiError as exc:
        if getattr(exc, "status_code", None) == 401:
            print(
                "[line] push failed: invalid LINE channel access token (HTTP 401). "
                "Check LINE_CHANNEL_ACCESS_TOKEN."
            )
        else:
            print(f"[line] push failed: {exc}")
        return False
    except RequestException as exc:
        print(f"[line] network error while pushing: {exc}")
        return False
    except Exception as exc:
        print(f"[line] unexpected push error: {exc}")
        return False


def seen_or_mark(message_id: str) -> bool:
    now = time.monotonic()
    with processed_lock:
        for key, exp in list(PROCESSED.items()):
            if exp < now:
                del PROCESSED[key]

        if message_id in PROCESSED:
            return True

        PROCESSED[message_id] = now + TTL_SECONDS
        return False


def _reject_non_text(event):
    safe_reply(
        event.reply_token,
        TextSendMessage(text="Distributed LLaMA only accepts text messages for inference"),
    )


@handler.add(MessageEvent, message=ImageMessage)
def on_image(event):
    _reject_non_text(event)


@handler.add(MessageEvent, message=VideoMessage)
def on_video(event):
    _reject_non_text(event)


@handler.add(MessageEvent, message=AudioMessage)
def on_audio(event):
    _reject_non_text(event)


@handler.add(MessageEvent, message=FileMessage)
def on_file(event):
    _reject_non_text(event)


@handler.add(MessageEvent, message=LocationMessage)
def on_location(event):
    _reject_non_text(event)


@handler.add(MessageEvent, message=StickerMessage)
def on_sticker(event):
    _reject_non_text(event)


def run_once(prompt: str) -> InferenceResult:
    workers = cluster.select_workers()
    task = InferenceTask(
        prompt=prompt,
        steps=INFERENCE_STEPS,
        workers=workers,
        cluster=cluster,
    )
    return task.run()


def run_with_recovery(user_id: str, prompt: str) -> InferenceResult:
    last_result: InferenceResult | None = None

    for attempt in range(MAX_RECOVERY_RETRIES + 1):
        workers = cluster.select_workers()
        node_count = 1 + len(workers)
        worker_names = ", ".join(worker.name for worker in workers) or "root only"

        if attempt == 0:
            safe_push(
                user_id,
                TextSendMessage(
                    text=f"Thinking... Using {node_count} node{'s' if node_count != 1 else ''}."
                ),
            )
        else:
            safe_push(
                user_id,
                TextSendMessage(
                    text=f"Retrying... Using {node_count} node{'s' if node_count != 1 else ''}."
                ),
            )

        task = InferenceTask(
            prompt=prompt,
            steps=INFERENCE_STEPS,
            workers=workers,
            cluster=cluster,
        )
        result = task.run()
        last_result = result

        if result.success:
            return result

        if result.reason != "worker_disconnected":
            return result

        failed = result.failed_worker or "a worker"
        safe_push(
            user_id,
            TextSendMessage(
                text="Worker failure detected. Reconfiguring and retrying this request."
            ),
        )

        # The heartbeat monitor normally marks the worker first. This short
        # delay gives the other monitors enough time to settle before choosing
        # the next legal 4/2/1-node topology.
        time.sleep(RECOVERY_SETTLE_SEC)

    assert last_result is not None
    return last_result


def _failure_text(result: InferenceResult) -> str:
    details = result.reason or "unknown error"
    if result.return_code is not None:
        details += f" (exit code {result.return_code})"
    return f"Inference failed: {details}"


def handle_waiting_users():
    global running

    while True:
        with waiting_lock:
            if user_waiting:
                user_id, prompt = user_waiting.popleft()
                running = True
            else:
                running = False
                return

        try:
            result = run_with_recovery(user_id, prompt)

            if result.success:
                text = result.output or "Inference completed, but no generated text was parsed."
                safe_push(user_id, TextSendMessage(text=text))
            else:
                safe_push(user_id, TextSendMessage(text=_failure_text(result)))
        except Exception as exc:
            print(f"[app] request failed: {exc}")
            safe_push(user_id, TextSendMessage(text=f"Inference failed: {exc}"))
        finally:
            user_modes[user_id] = "idle"


def enqueue_inference(user_id: str, prompt: str):
    global running

    with waiting_lock:
        user_waiting.append((user_id, prompt))
        should_start = len(user_waiting) == 1 and not running
        if should_start:
            # Mark it before starting the thread to remove a small race where a
            # second webhook could start another queue consumer.
            running = True
            threading.Thread(target=handle_waiting_users, daemon=True).start()


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    msg_id = event.message.id
    if seen_or_mark(msg_id):
        return

    user_id = event.source.user_id
    text = event.message.text.strip()
    command = text.lower()

    # Status is available regardless of the user's current inference state.
    if command == "status":
        safe_reply(event.reply_token, TextSendMessage(text=cluster.format_status()))
        return

    if user_id not in user_modes:
        user_modes[user_id] = "idle"

    if user_modes[user_id] == "idle":
        if command == "inference":
            user_modes[user_id] = "awaiting_inference_prompt"
            safe_reply(
                event.reply_token,
                TextSendMessage(text="Please enter your prompt for inference (English only):"),
            )
        else:
            safe_reply(
                event.reply_token,
                TextSendMessage(text='Unknown command. Use "inference" or "status".'),
            )
        return

    if user_modes[user_id] == "awaiting_inference_prompt":
        user_modes[user_id] = "inference"
        safe_reply(
            event.reply_token,
            TextSendMessage(text="Your inference request has been queued."),
        )
        enqueue_inference(user_id, text)
        return

    if user_modes[user_id] == "inference":
        safe_reply(
            event.reply_token,
            TextSendMessage(text="Model is running. Use status to check the cluster."),
        )


@app.route("/", methods=["POST"])
def linechatbot():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")

    if not signature:
        abort(400, "Missing X-Line-Signature")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    except Exception as exc:
        print(f"[line] webhook handler error: {exc}")

    return "OK", 200


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        cluster.stop()
