"""Minimal OpenAI-compatible upstream used by vllm-sr CLI integration tests."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        print(self.path, flush=True)
        body = (
            b'{"id":"mock","object":"chat.completion",'
            b'"model":"test-model","choices":[{"index":0,'
            b'"message":{"role":"assistant","content":"ok"},'
            b'"finish_reason":"stop"}]}'
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


ThreadingHTTPServer(("0.0.0.0", 18080), Handler).serve_forever()
