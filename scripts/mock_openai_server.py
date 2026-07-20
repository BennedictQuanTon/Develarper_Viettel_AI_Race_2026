#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock server for local scaffold smoke (no GPU).

Usage:
  python scripts/mock_openai_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODEL_ID = "mock-model"
REPLY = "Hello from the Develarper mock server."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter
        print(f"[mock] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            payload = {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return

        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return

        model = req.get("model") or MODEL_ID
        stream = bool(req.get("stream"))

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Fake TTFT
            time.sleep(0.02)
            for i, ch in enumerate(REPLY):
                chunk = {
                    "id": "mock-chat",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ch} if i == 0 else {"content": ch},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)

            done = {
                "id": "mock-chat",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        payload = {
            "id": "mock-chat",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": len(REPLY.split()), "total_tokens": 8 + len(REPLY.split())},
        }
        self._send(200, json.dumps(payload).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock] listening on http://{args.host}:{args.port}  model={MODEL_ID}")
    server.serve_forever()


if __name__ == "__main__":
    main()
