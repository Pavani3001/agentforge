"""Minimal web UI and JSON API for AgentForge."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentforge.llm import LLMError
from agentforge.pipeline import AgentForgePipeline, PipelineError

STATIC_DIR = Path(__file__).parent / "web"


class Handler(BaseHTTPRequestHandler):
    pipeline = AgentForgePipeline

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        content = (STATIC_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            criteria = payload.get("criteria", [])
            if isinstance(criteria, str):
                criteria = criteria.splitlines()
            result = self.pipeline().run(payload.get("objective", ""), criteria)
            self._json(200, result.to_dict())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Invalid request: {exc}"})
        except (LLMError, PipelineError) as exc:
            self._json(422, {"error": str(exc)})

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("AgentForge running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
