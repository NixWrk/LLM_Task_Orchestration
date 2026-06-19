"""Minimal host-side bridge that exposes the local `lms` CLI over HTTP.

The orchestrator's lifecycle service runs inside a container and owns model
loading, but `lms` is a host-bound binary tied to the local LM Studio app and
cannot be invoked from the container (no `--host`, not installable in-image).

This bridge runs on the host and executes whitelisted `lms` subcommands on
request. A tiny `lms` shim inside the lifecycle container forwards every
`lms ...` invocation here, so the orchestrator keeps calling `lms load
--context-length ...` exactly as designed -- the load logic is unchanged.

Run on the host:

    LMS_BINARY="C:\\Users\\<you>\\.lmstudio\\bin\\lms.exe" \\
    python services/lms_bridge/lms_bridge.py --host 0.0.0.0 --port 4399
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Only these first-argument subcommands may be executed through the bridge.
ALLOWED_SUBCOMMANDS = {"load", "unload", "ps", "ls", "server", "status", "version", "--version"}
DEFAULT_TIMEOUT_SECONDS = 1800


def resolve_lms_binary() -> str:
    configured = os.environ.get("LMS_BINARY", "").strip()
    if configured:
        return configured
    found = shutil.which("lms") or shutil.which("lms.exe")
    return found or "lms"


LMS_BINARY = resolve_lms_binary()


def run_lms(args: list[str], timeout_s: int) -> dict[str, object]:
    if not args:
        return {"returncode": 2, "stdout": "", "stderr": "lms bridge: empty args"}
    if args[0] not in ALLOWED_SUBCOMMANDS:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": f"lms bridge: subcommand {args[0]!r} not allowed",
        }
    try:
        completed = subprocess.run(
            [LMS_BINARY, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": f"lms bridge: timed out after {timeout_s}s",
        }
    except FileNotFoundError:
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": f"lms bridge: lms binary not found at {LMS_BINARY!r}",
        }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            self._send(200, {"status": "ok", "lms_binary": LMS_BINARY})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/run":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, {"returncode": 2, "stdout": "", "stderr": "invalid json"})
            return
        args = payload.get("args") or []
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            self._send(400, {"returncode": 2, "stdout": "", "stderr": "args must be string[]"})
            return
        timeout_s = int(payload.get("timeout") or DEFAULT_TIMEOUT_SECONDS)
        self._send(200, run_lms(args, timeout_s))

    def log_message(self, fmt: str, *args: object) -> None:
        # Quiet by default; the orchestrator already logs load decisions.
        print("lms-bridge:", fmt % args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Host-side lms CLI HTTP bridge.")
    parser.add_argument("--host", default=os.environ.get("LMS_BRIDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LMS_BRIDGE_PORT", "4399")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"lms bridge listening on {args.host}:{args.port} (lms={LMS_BINARY})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
