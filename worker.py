from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time


HEARTBEAT_HOST = os.environ.get("HEARTBEAT_HOST", "0.0.0.0")
HEARTBEAT_PORT = int(os.environ.get("HEARTBEAT_PORT", "9800"))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "0.25"))

DLLAMA_BIN = os.environ.get("DLLAMA_BIN", "./distributed-llama/dllama")
DLLAMA_WORKER_PORT = int(os.environ.get("DLLAMA_WORKER_PORT", "9999"))
DLLAMA_WORKER_THREADS = int(os.environ.get("DLLAMA_WORKER_THREADS", "4"))
DLLAMA_RESTART_DELAY = float(os.environ.get("DLLAMA_RESTART_DELAY", "0.5"))
DLLAMA_READY_DELAY = float(os.environ.get("DLLAMA_READY_DELAY", "0.35"))


class WorkerService:
    """Supervises a long-running dllama worker and exposes its health to the root."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.proc_lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.supervisor_thread: threading.Thread | None = None
        self.heartbeat_thread: threading.Thread | None = None

    def _dllama_command(self) -> list[str]:
        return [
            DLLAMA_BIN,
            "worker",
            "--port",
            str(DLLAMA_WORKER_PORT),
            "--nthreads",
            str(DLLAMA_WORKER_THREADS),
        ]

    def _stream_output(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break

            text = line.rstrip()
            print(f"[dllama] {text}")

            # Root 連線斷掉時先把 READY 清掉，不用等到 dllama process 完全結束。
            lowered = text.lower()
            if (
                "network is closed" in lowered
                or "network error" in lowered
                or "critical error" in lowered
            ):
                self.ready_event.clear()

    def _supervise_dllama(self) -> None:
        while not self.stop_event.is_set():
            cmd = self._dllama_command()
            print("[worker] starting:", " ".join(cmd))

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except Exception as exc:
                self.ready_event.clear()
                print(f"[worker] failed to start dllama: {exc}")
                self.stop_event.wait(DLLAMA_RESTART_DELAY)
                continue

            with self.proc_lock:
                self.proc = proc

            output_thread = threading.Thread(
                target=self._stream_output,
                args=(proc,),
                daemon=True,
                name="dllama-output",
            )
            output_thread.start()

            # Give dllama a short initialization window. Do not probe port 9999:
            # connecting to it would be interpreted as a root-node connection.
            if not self.stop_event.wait(DLLAMA_READY_DELAY) and proc.poll() is None:
                self.ready_event.set()
                print("[worker] dllama is ready")

            return_code = proc.wait()
            self.ready_event.clear()

            with self.proc_lock:
                if self.proc is proc:
                    self.proc = None

            if self.stop_event.is_set():
                break

            print(f"[worker] dllama exited with code {return_code}; restarting")
            self.stop_event.wait(DLLAMA_RESTART_DELAY)

    def _serve_heartbeat_client(self, client: socket.socket, addr) -> None:
        print(f"[worker] heartbeat client connected from {addr}")
        client.settimeout(max(2.0, HEARTBEAT_INTERVAL * 3))
        try:
            while not self.stop_event.is_set():
                state = "READY" if self.ready_event.is_set() else "NOT_READY"
                client.sendall(f"HEARTBEAT {state}\n".encode("utf-8"))
                if self.stop_event.wait(HEARTBEAT_INTERVAL):
                    break
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass
            print("[worker] heartbeat client disconnected")

    def _heartbeat_server(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HEARTBEAT_HOST, HEARTBEAT_PORT))
        server.listen(4)
        server.settimeout(0.5)
        print(f"[worker] heartbeat server listening on {HEARTBEAT_HOST}:{HEARTBEAT_PORT}")

        try:
            while not self.stop_event.is_set():
                try:
                    client, addr = server.accept()
                except socket.timeout:
                    continue
                self._serve_heartbeat_client(client, addr)
        finally:
            server.close()

    def start(self) -> None:
        self.supervisor_thread = threading.Thread(
            target=self._supervise_dllama,
            daemon=True,
            name="dllama-supervisor",
        )
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_server,
            daemon=True,
            name="heartbeat-server",
        )
        self.supervisor_thread.start()
        self.heartbeat_thread.start()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.ready_event.clear()

        with self.proc_lock:
            proc = self.proc

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        for thread in (self.supervisor_thread, self.heartbeat_thread):
            if thread is not None:
                thread.join(timeout=2.0)


def main() -> None:
    service = WorkerService()

    def _shutdown(_signum=None, _frame=None):
        service.close()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    service.start()
    try:
        while not service.stop_event.wait(1.0):
            pass
    finally:
        service.close()


if __name__ == "__main__":
    main()
