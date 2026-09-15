from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import subprocess
import threading
import time

from cluster import ClusterManager, WorkerSpec


DLLAMA_BIN = os.environ.get("DLLAMA_BIN", "./distributed-llama/dllama")
MODEL_PATH = os.environ.get(
    "DLLAMA_MODEL_PATH",
    "distributed-llama/models/dllama_model_llama3_2_3b_instruct_q40.m",
)
TOKENIZER_PATH = os.environ.get(
    "DLLAMA_TOKENIZER_PATH",
    "distributed-llama/models/dllama_tokenizer_llama3_2_3b_instruct_q40.t",
)
ROOT_THREADS = int(os.environ.get("DLLAMA_ROOT_THREADS", "4"))
MAX_SEQ_LEN = int(os.environ.get("DLLAMA_MAX_SEQ_LEN", "512"))
INFERENCE_TIMEOUT = float(os.environ.get("DLLAMA_INFERENCE_TIMEOUT", "120"))


@dataclass
class InferenceResult:
    success: bool
    output: str = ""
    reason: str | None = None
    failed_worker: str | None = None
    workers: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return 1 + len(self.workers)


class InferenceTask:
    def __init__(
        self,
        prompt: str,
        steps: int = 30,
        workers: list[WorkerSpec] | None = None,
        cluster: ClusterManager | None = None,
        timeout: float = INFERENCE_TIMEOUT,
    ):
        self.prompt = prompt or ""
        self.steps = steps
        self.workers = list(workers or [])
        self.cluster = cluster
        self.timeout = timeout

        self.proc: subprocess.Popen | None = None
        self.stop_event = threading.Event()
        self.t_out: threading.Thread | None = None
        self.t_err: threading.Thread | None = None

    def _build_command(self) -> list[str]:
        cmd = [
            DLLAMA_BIN,
            "inference",
            "--buffer-float-type",
            "q80",
            "--prompt",
            self.prompt,
            "--max-seq-len",
            str(MAX_SEQ_LEN),
            "--steps",
            str(self.steps),
            "--nthreads",
            str(ROOT_THREADS),
            "--model",
            MODEL_PATH,
            "--tokenizer",
            TOKENIZER_PATH,
        ]

        if self.workers:
            cmd.append("--workers")
            cmd.extend(worker.inference_address for worker in self.workers)

        return cmd

    def _terminate_process(self) -> None:
        self.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        for thread in (self.t_out, self.t_err):
            if thread is not None:
                thread.join(timeout=0.5)

    def _selected_worker_failure(self) -> str | None:
        if self.cluster is None:
            return None
        for worker in self.workers:
            snapshot = self.cluster.get_worker(worker.name)
            if snapshot is None or not snapshot.alive or not snapshot.ready:
                return worker.name
        return None

    @staticmethod
    def _looks_like_network_error(lines: list[str]) -> bool:
        text = "\n".join(lines).lower()
        patterns = (
            "network error",
            "network is closed",
            "connection error",
            "connection refused",
            "broken pipe",
            "socket error",
            "failed to connect",
        )
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _extract_pred_piece(line: str) -> str | None:
        # Current Distributed Llama benchmark lines look like:
        # "Pred ... | <decoded token>". Remove only the formatting space
        # after '|', preserving a genuine leading space in the decoded token.
        if "Pred" not in line or "|" not in line:
            return None
        piece = line.rsplit("|", 1)[1]
        if piece.startswith(" "):
            piece = piece[1:]
        return piece

    def run(self) -> InferenceResult:
        start = time.monotonic()
        worker_names = [worker.name for worker in self.workers]
        cmd = self._build_command()
        print(f"[inference] topology={1 + len(self.workers)} nodes")
        print("[inference] command:", " ".join(cmd[:2] + ["..."]))

        line_q: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        generated_parts: list[str] = []

        def reader(stream, source: str) -> None:
            try:
                for raw in iter(stream.readline, ""):
                    if self.stop_event.is_set():
                        break
                    line_q.put((source, raw.rstrip("\n")))
            finally:
                line_q.put((source, None))

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as exc:
            return InferenceResult(
                success=False,
                reason=f"failed_to_start: {exc}",
                workers=worker_names,
                elapsed=time.monotonic() - start,
            )

        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self.t_out = threading.Thread(target=reader, args=(self.proc.stdout, "stdout"), daemon=True)
        self.t_err = threading.Thread(target=reader, args=(self.proc.stderr, "stderr"), daemon=True)
        self.t_out.start()
        self.t_err.start()

        closed_streams: set[str] = set()

        try:
            while True:
                failed_worker = self._selected_worker_failure()
                if failed_worker is not None:
                    elapsed = time.monotonic() - start
                    self._terminate_process()
                    return InferenceResult(
                        success=False,
                        reason="worker_disconnected",
                        failed_worker=failed_worker,
                        workers=worker_names,
                        elapsed=elapsed,
                        logs=stdout_lines + stderr_lines,
                    )

                if time.monotonic() - start > self.timeout:
                    elapsed = time.monotonic() - start
                    self._terminate_process()
                    return InferenceResult(
                        success=False,
                        reason="inference_timeout",
                        workers=worker_names,
                        elapsed=elapsed,
                        logs=stdout_lines + stderr_lines,
                    )

                try:
                    source, line = line_q.get(timeout=0.2)
                except queue.Empty:
                    if self.proc.poll() is not None and len(closed_streams) == 2 and line_q.empty():
                        break
                    continue

                if line is None:
                    closed_streams.add(source)
                    if self.proc.poll() is not None and len(closed_streams) == 2 and line_q.empty():
                        break
                    continue

                if source == "stdout":
                    stdout_lines.append(line)
                    piece = self._extract_pred_piece(line)
                    if piece is not None:
                        generated_parts.append(piece)
                else:
                    stderr_lines.append(line)

            return_code = self.proc.wait(timeout=1.0)
            elapsed = time.monotonic() - start
            all_logs = stdout_lines + stderr_lines

            if return_code != 0:
                failed_worker = self._selected_worker_failure()
                if self.workers and (failed_worker is not None or self._looks_like_network_error(all_logs)):
                    return InferenceResult(
                        success=False,
                        reason="worker_disconnected",
                        failed_worker=failed_worker,
                        workers=worker_names,
                        elapsed=elapsed,
                        return_code=return_code,
                        logs=all_logs,
                    )

                return InferenceResult(
                    success=False,
                    reason="process_error",
                    workers=worker_names,
                    elapsed=elapsed,
                    return_code=return_code,
                    logs=all_logs,
                )

            output = "".join(generated_parts).strip()
            if not output:
                # Compatibility fallback for a locally modified dllama build that
                # prints an explicit "Output Message:" line.
                for line in reversed(stdout_lines):
                    if "Output Message:" in line:
                        output = line.split("Output Message:", 1)[1].strip()
                        break

            return InferenceResult(
                success=True,
                output=output,
                workers=worker_names,
                elapsed=elapsed,
                return_code=return_code,
                logs=all_logs,
            )
        finally:
            self.stop_event.set()
            for thread in (self.t_out, self.t_err):
                if thread is not None:
                    thread.join(timeout=0.5)

    # Keep the old method name so any existing local scripts do not break.
    def _run_inference(self) -> InferenceResult:
        return self.run()
