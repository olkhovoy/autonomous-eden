#!/usr/bin/env python3
"""
Soul Monitor server with API for thoughts and interaction.
"""

import argparse
import json
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import requests

# Configurable paths
THOUGHTS_LOG = os.getenv("THOUGHTS_LOG", "logs/inner_monologue.jsonl")
MEMORY_ENDPOINT = os.getenv("MEMORY_ENDPOINT", "http://localhost:8087")
SOUL_ID = os.getenv("SOUL_ID", "eve")


class SoulMonitorHandler(SimpleHTTPRequestHandler):
    """Handler with API endpoints for thoughts and interaction."""
    
    base_dir = "."
    
    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
    
    def _json_response(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
    
    def end_headers(self):
        self._cors_headers()
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API: Get latest thoughts
        if parsed.path == "/api/thoughts":
            return self._get_thoughts(parsed.query)
        
        # Serve static files
        return super().do_GET()
    
    def do_POST(self):
        parsed = urlparse(self.path)
        
        # API: Send message to EVE
        if parsed.path == "/api/message":
            return self._send_message()
        
        self._json_response(404, {"error": "not found"})
    
    def _get_thoughts(self, query_string: str):
        """Get latest thoughts from log file."""
        params = parse_qs(query_string)
        limit = int(params.get('limit', ['10'])[0])
        
        thoughts_path = os.path.join(self.base_dir, "..", "..", THOUGHTS_LOG)
        thoughts = []
        
        try:
            if os.path.exists(thoughts_path):
                with open(thoughts_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            t = json.loads(line.strip())
                            thoughts.append({
                                "ts": t.get("ts", 0),
                                "thought": t.get("thought", "")
                            })
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            return self._json_response(500, {"error": str(e)})
        
        self._json_response(200, {"thoughts": thoughts})
    
    def _send_message(self):
        """Send message to EVE (stores in memory and triggers thought)."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            message = data.get("message", "").strip()
            
            if not message:
                return self._json_response(400, {"error": "message required"})
            
            # Store in EVE's memory
            memory_payload = {
                "soul_id": SOUL_ID,
                "text": f"[USER INPUT] {message}",
                "tags": ["user_input", "interaction"],
                "meta": {"type": "user_message", "ts": time.time()}
            }
            
            resp = requests.post(
                f"{MEMORY_ENDPOINT}/memories/ingest",
                json=memory_payload,
                timeout=10
            )
            
            if resp.status_code == 200:
                self._json_response(200, {
                    "ok": True,
                    "message": "Message sent to EVE",
                    "stored": True
                })
            else:
                self._json_response(500, {
                    "error": "Failed to store message",
                    "details": resp.text
                })
                
        except Exception as e:
            self._json_response(500, {"error": str(e)})


def main():
    global THOUGHTS_LOG, MEMORY_ENDPOINT
    
    parser = argparse.ArgumentParser(description="Soul Monitor web server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--thoughts-log", type=str, default=THOUGHTS_LOG)
    parser.add_argument("--memory-endpoint", type=str, default=MEMORY_ENDPOINT)
    args = parser.parse_args()
    
    THOUGHTS_LOG = args.thoughts_log
    MEMORY_ENDPOINT = args.memory_endpoint
    
    SoulMonitorHandler.base_dir = args.dir
    os.chdir(args.dir)
    
    server = HTTPServer(("0.0.0.0", args.port), SoulMonitorHandler)
    print(f"Soul Monitor serving on http://localhost:{args.port}", flush=True)
    print(f"  /api/thoughts - get EVE's thoughts", flush=True)
    print(f"  /api/message  - send message to EVE", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
