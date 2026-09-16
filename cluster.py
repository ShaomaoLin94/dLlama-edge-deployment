from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import threading
import time
from typing import Iterable


DEFAULT_INFERENCE_PORT = 9999
DEFAULT_HEARTBEAT_PORT = 9800
DEFAULT_HEARTBEAT_TIMEOUT = float(os.environ.get("DLLAMA_HEARTBEAT_TIMEOUT", "0.9"))
DEFAULT_RECONNECT_INTERVAL = float(os.environ.get("DLLAMA_RECONNECT_INTERVAL", "0.2"))
HEARTBEAT_SOCKET_TIMEOUT = float(os.environ.get("DLLAMA_HEARTBEAT_SOCKET_TIMEOUT", "0.25"))


def _detect_root_ip(worker_specs: list["WorkerSpec"]) -> str:
    explicit = os.environ.get("DLLAMA_ROOT_IP")
    if explicit:
        return explicit.strip()

    # Prefer the local address that Linux would use to reach a configured worker.
    if worker_specs:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((worker_specs[0].host, worker_specs[0].heartbeat_port))
            return sock.getsockname()[0]
        except OSError:
            pass
        finally:
            sock.close()

    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    host: str
    inference_port: int = DEFAULT_INFERENCE_PORT
    heartbeat_port: int = DEFAULT_HEARTBEAT_PORT

    @property
    def inference_address(self) -> str:
        return f"{self.host}:{self.inference_port}"


@dataclass
class WorkerSnapshot:
    name: str
    host: str
    inference_port: int
    heartbeat_port: int
    alive: bool
    ready: bool
    last_seen_age: float | None
    reason: str


class _WorkerRuntime:
    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.alive = False
        self.ready = False
        self.last_seen = 0.0
        self.reason = "not checked yet"
        self.generation = 0


class ClusterManager:
    """Continuously monitors worker heartbeat services and selects a legal dllama topology."""

    def __init__(
        self,
        workers: Iterable[WorkerSpec],
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
    ):
        worker_specs = list(workers)
        self._workers = {w.name: _WorkerRuntime(w) for w in worker_specs}
        self.root_name = os.environ.get("DLLAMA_ROOT_NAME", socket.gethostname()).strip() or "root"
        self.root_host = _detect_root_ip(worker_specs)
        self.heartbeat_timeout = heartbeat_timeout
        self.reconnect_interval = reconnect_interval
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True

        for name in self._workers:
            thread = threading.Thread(
                target=self._monitor_worker,
                args=(name,),
                daemon=True,
                name=f"heartbeat-{name}",
            )
            self._threads.append(thread)
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def _set_state(self, name: str, *, alive: bool, ready: bool, reason: str, seen: bool) -> None:
        now = time.monotonic()
        with self._lock:
            runtime = self._workers[name]
            changed = runtime.alive != alive or runtime.ready != ready
            runtime.alive = alive
            runtime.ready = ready
            runtime.reason = reason
            if seen:
                runtime.last_seen = now
            if changed:
                runtime.generation += 1
                print(
                    f"[cluster] {name}: "
                    f"{'READY' if alive and ready else 'UNAVAILABLE'} ({reason})"
                )

    def mark_unavailable(self, name: str, reason: str) -> None:
        if name not in self._workers:
            return
        self._set_state(name, alive=False, ready=False, reason=reason, seen=False)

    def _monitor_worker(self, name: str) -> None:
        spec = self._workers[name].spec

        while not self._stop_event.is_set():
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection(
                    (spec.host, spec.heartbeat_port),
                    timeout=min(1.0, self.heartbeat_timeout),
                )
                sock.settimeout(HEARTBEAT_SOCKET_TIMEOUT)
                buffer = b""
                last_rx = time.monotonic()

                while not self._stop_event.is_set():
                    try:
                        data = sock.recv(1024)
                        if not data:
                            raise ConnectionError("heartbeat socket closed")
                        last_rx = time.monotonic()
                        buffer += data

                        while b"\n" in buffer:
                            raw, buffer = buffer.split(b"\n", 1)
                            message = raw.decode("utf-8", errors="replace").strip().upper()
                            if not message:
                                continue

                            if message in {"READY", "HEARTBEAT READY", "HEARTBEAT"}:
                                self._set_state(
                                    name,
                                    alive=True,
                                    ready=True,
                                    reason="heartbeat ready",
                                    seen=True,
                                )
                            elif message in {"NOT_READY", "HEARTBEAT NOT_READY"}:
                                self._set_state(
                                    name,
                                    alive=False,
                                    ready=False,
                                    reason="worker process not ready",
                                    seen=True,
                                )
                            else:
                                # Unknown heartbeat payload still proves the host is reachable,
                                # but do not schedule inference on it.
                                self._set_state(
                                    name,
                                    alive=False,
                                    ready=False,
                                    reason=f"unknown heartbeat: {message}",
                                    seen=True,
                                )

                    except socket.timeout:
                        if time.monotonic() - last_rx > self.heartbeat_timeout:
                            raise TimeoutError("heartbeat timeout")

            except Exception as exc:
                self._set_state(
                    name,
                    alive=False,
                    ready=False,
                    reason=str(exc),
                    seen=False,
                )
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

            if not self._stop_event.wait(self.reconnect_interval):
                continue

    def get_worker(self, name: str) -> WorkerSnapshot | None:
        now = time.monotonic()
        with self._lock:
            runtime = self._workers.get(name)
            if runtime is None:
                return None
            age = None if runtime.last_seen == 0 else max(0.0, now - runtime.last_seen)
            return WorkerSnapshot(
                name=runtime.spec.name,
                host=runtime.spec.host,
                inference_port=runtime.spec.inference_port,
                heartbeat_port=runtime.spec.heartbeat_port,
                alive=runtime.alive,
                ready=runtime.ready,
                last_seen_age=age,
                reason=runtime.reason,
            )

    def snapshots(self) -> list[WorkerSnapshot]:
        now = time.monotonic()
        result: list[WorkerSnapshot] = []
        with self._lock:
            for runtime in self._workers.values():
                age = None if runtime.last_seen == 0 else max(0.0, now - runtime.last_seen)
                result.append(
                    WorkerSnapshot(
                        name=runtime.spec.name,
                        host=runtime.spec.host,
                        inference_port=runtime.spec.inference_port,
                        heartbeat_port=runtime.spec.heartbeat_port,
                        alive=runtime.alive,
                        ready=runtime.ready,
                        last_seen_age=age,
                        reason=runtime.reason,
                    )
                )
        return result

    def alive_workers(self) -> list[WorkerSpec]:
        with self._lock:
            return [
                runtime.spec
                for runtime in self._workers.values()
                if runtime.alive and runtime.ready
            ]

    def select_workers(self) -> list[WorkerSpec]:
        """Select the largest legal dllama topology for a 1/2/4-node Pi cluster.

        The root is itself one node. Therefore:
        - 3 healthy workers -> root + 3 workers = 4 nodes
        - 1 or 2 healthy workers -> root + 1 worker = 2 nodes
        - 0 healthy workers -> root only = 1 node
        """
        alive = self.alive_workers()
        if len(alive) >= 3:
            return alive[:3]
        if len(alive) >= 1:
            return alive[:1]
        return []

    def topology_nodes(self) -> int:
        return 1 + len(self.select_workers())

    def wait_for_recovery(
        self,
        timeout: float = 4.0,
        stable_for: float = 0.4,
    ) -> list[WorkerSpec]:
        """Wait for a usable topology before retrying inference."""
        deadline = time.monotonic() + timeout
        stable_names: tuple[str, ...] | None = None
        stable_since: float | None = None

        while time.monotonic() < deadline and not self._stop_event.is_set():
            selected = self.select_workers()
            names = tuple(worker.name for worker in selected)

            # 至少等到一個 worker 回來，避免太早退成 root-only
            if names:
                if names != stable_names:
                    stable_names = names
                    stable_since = time.monotonic()
                elif stable_since is not None and time.monotonic() - stable_since >= stable_for:
                    print(f"[cluster] recovery topology ready: {1 + len(selected)} nodes")
                    return selected
            else:
                stable_names = None
                stable_since = None

            self._stop_event.wait(0.05)

        # 超時後才使用目前能用的 topology，這時才可能是 root-only
        selected = self.select_workers()
        print(f"[cluster] recovery wait timeout, using {1 + len(selected)} nodes")
        return selected

    def format_status(self) -> str:
        snapshots = self.snapshots()
        selected = {worker.name for worker in self.select_workers()}
        node_count = self.topology_nodes()

        lines = [
            f"Cluster topology: {node_count} node{'s' if node_count != 1 else ''}",
            "",
            "Root:",
            f"{self.root_name} ({self.root_host}): ACTIVE",
        ]

        if not snapshots:
            lines.extend(["", "Workers:", "No workers configured"])
            return "\n".join(lines)

        lines.extend(["", "Workers:"])
        for worker in snapshots:
            age = "never" if worker.last_seen_age is None else f"{worker.last_seen_age:.1f}s ago"

            if worker.alive and worker.ready:
                state = "ACTIVE" if worker.name in selected else "READY (standby)"
            else:
                state = "DOWN"

            lines.append(f"{worker.name} ({worker.host}): {state}, heartbeat {age}")

        return "\n".join(lines)


def load_worker_specs(value: str | None = None) -> list[WorkerSpec]:
    """Load workers from DLLAMA_WORKERS.

    Supported entries (comma separated):
      192.168.0.15
      pi2=192.168.0.12
      pi2=192.168.0.12:9999:9800

    For backward compatibility, if DLLAMA_WORKERS is unset the old single
    worker address 192.168.0.15 is used.
    """
    raw = value if value is not None else os.environ.get("DLLAMA_WORKERS", "192.168.0.15")
    raw = raw.strip()
    if not raw:
        return []

    specs: list[WorkerSpec] = []
    for index, entry in enumerate(raw.split(","), start=1):
        entry = entry.strip()
        if not entry:
            continue

        if "=" in entry:
            name, address = entry.split("=", 1)
            name = name.strip()
            address = address.strip()
        else:
            name = f"worker{index}"
            address = entry

        parts = address.split(":")
        host = parts[0].strip()
        inference_port = int(parts[1]) if len(parts) >= 2 and parts[1] else DEFAULT_INFERENCE_PORT
        heartbeat_port = int(parts[2]) if len(parts) >= 3 and parts[2] else DEFAULT_HEARTBEAT_PORT

        if not name or not host:
            raise ValueError(f"Invalid worker entry: {entry}")

        specs.append(
            WorkerSpec(
                name=name,
                host=host,
                inference_port=inference_port,
                heartbeat_port=heartbeat_port,
            )
        )

    return specs
