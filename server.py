from __future__ import annotations

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

PORT = 8000
WEB_DIR = Path(__file__).resolve().parent / "web"


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[Server] {self.address_string()} - {args[0]} {args[1]}\n")


def start_server(port: int = PORT, auto_open: bool = True) -> None:
    # Verify data files exist
    recs_json = WEB_DIR / "data" / "recommendations.json"
    if not recs_json.exists():
        print("[!] Generating web dataset from pipeline outputs...")
        from scripts.prepare_web_data import prepare_web_data
        prepare_web_data()

    # Try binding to port, fallback if busy
    current_port = port
    max_attempts = 10
    httpd = None

    for attempt in range(max_attempts):
        try:
            httpd = socketserver.TCPServer(("", current_port), CustomHTTPRequestHandler)
            break
        except OSError:
            current_port += 1

    if httpd is None:
        print(f"[ERROR] Could not bind to any port from {port} to {port + max_attempts}.")
        sys.exit(1)

    url = f"http://localhost:{current_port}"
    print("=" * 70)
    print("  [>] Faculty-Course Dual-Layer Recommender Web Dashboard (M4-v2)")
    print(f"  [*] Running locally at: {url}")
    print("  [*] Serving directory: web/")
    print("  [*] Press Ctrl+C to stop the server anytime.")
    print("=" * 70)

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down cleanly. Goodbye!")
        httpd.server_close()


if __name__ == "__main__":
    auto_launch = "--no-browser" not in sys.argv
    start_server(auto_open=auto_launch)
