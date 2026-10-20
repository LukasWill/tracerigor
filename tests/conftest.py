import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

RUBRICS = ("observation_grounding", "action_coherence", "temporal_consistency")


@pytest.fixture
def stub_judge_server():
    """Local OpenAI-compatible endpoint that answers YES on every rubric.

    Yields ``(base_url, requests)``; ``requests`` collects each request body.
    """
    requests = []
    reply = json.dumps({name: {"yes_no": "YES", "evidence": "stub"} for name in RUBRICS})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(json.loads(body))
            payload = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "stub",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": reply},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
