#!/usr/bin/env python3
"""
Web-based chat client for hallucination detection demo.

Provides a visual browser interface showing the tool calling workflow.

Usage:
    python web_client.py --port 8888 --router-url http://localhost:8801 --search-url http://localhost:8003
"""

import argparse
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
import os

# HTML template for the chat interface
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hallucination Gatekeeper</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'SF Pro Display', sans-serif;
               background: #fafafa; min-height: 100vh; color: #1a1a1a; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px 30px; }

        /* Header */
        .header { text-align: center; padding: 30px 0 20px; border-bottom: 1px solid #e0e0e0; margin-bottom: 30px; }
        h1 { font-size: 32px; font-weight: 700; color: #000; letter-spacing: -0.5px; }
        .subtitle { color: #666; margin-top: 8px; font-size: 14px; font-weight: 400; }

        /* Main Layout - Detection result prominently on top right */
        .main-layout { display: grid; grid-template-columns: 1fr 380px; grid-template-rows: auto 1fr; gap: 24px; }
        @media (max-width: 1000px) { .main-layout { grid-template-columns: 1fr; } }

        /* Detection Result Card - Most prominent */
        .detection-card { grid-column: 2; grid-row: 1; background: #fff; border-radius: 12px;
                          border: 2px solid #e0e0e0; padding: 20px; min-height: 140px; }
        .detection-card.detected { border-color: #000; background: #000; color: #fff; }
        .detection-card.safe { border-color: #000; }
        .detection-card.warning { border-color: #cc6600; background: #fff8f0; }
        .detection-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 12px; }
        .detection-card.detected .detection-title { color: #888; }
        .detection-card.warning .detection-title { color: #cc6600; }
        .detection-status { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
        .detection-status .icon { font-size: 32px; }
        .detection-card.warning .detection-status { color: #cc6600; }
        .detection-detail { margin-top: 12px; font-size: 13px; color: #666; }
        .detection-card.detected .detection-detail { color: #aaa; }
        .detection-card.warning .detection-detail { color: #996633; }
        .flagged-items { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
        .flagged-item { background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 4px; font-size: 12px; font-family: monospace; }
        .detection-card:not(.detected) .flagged-item { background: #f0f0f0; }
        .detection-card.warning .flagged-item { background: #ffe8cc; color: #663300; }

        /* Chat Panel */
        .chat-panel { grid-column: 1; grid-row: 1 / 3; background: #fff; border-radius: 12px;
                      border: 1px solid #e0e0e0; display: flex; flex-direction: column; }
        .panel-header { padding: 16px 20px; border-bottom: 1px solid #e0e0e0; font-weight: 600; font-size: 14px;
                        display: flex; justify-content: space-between; align-items: center; }
        .new-chat-btn { width: 28px; height: 28px; border-radius: 6px; border: 1px solid #ddd; background: #fff;
                        cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center;
                        color: #666; transition: all 0.2s; }
        .new-chat-btn:hover { background: #f5f5f5; border-color: #000; color: #000; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 20px; min-height: 400px; max-height: 500px; }
        .chat-input-area { padding: 16px; border-top: 1px solid #e0e0e0; display: flex; gap: 12px; align-items: center; }
        .chat-input-area input[type="text"] { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #ddd;
                                  background: #fff; color: #000; font-size: 14px; }
        .chat-input-area input[type="text"]:focus { outline: none; border-color: #000; }
        .chat-input-area button { padding: 12px 28px; border-radius: 8px; border: none;
                                   background: #000; color: #fff; cursor: pointer; font-weight: 600;
                                   font-size: 14px; transition: all 0.2s; }
        .chat-input-area button:hover { background: #333; }
        .chat-input-area button:disabled { background: #ccc; cursor: not-allowed; }
        /* WebSearch Checkbox */
        .websearch-toggle { display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
        .websearch-toggle input[type="checkbox"] { width: 18px; height: 18px; cursor: pointer; accent-color: #000; }
        .websearch-toggle span { font-size: 14px; font-weight: 500; color: #333; }

        /* Messages */
        .message { margin-bottom: 16px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { text-align: right; }
        .message.user .bubble { background: #000; color: #fff; }
        .message.assistant .bubble { background: #f5f5f5; color: #000; }
        .message.error .bubble { background: #fff0f0; border: 1px solid #ffccc; color: #c00; }
        .bubble { display: inline-block; padding: 12px 16px; border-radius: 12px; max-width: 85%; text-align: left; font-size: 14px; line-height: 1.5; }
        .message-label { font-size: 10px; color: #999; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

        /* Workflow Panel */
        .workflow-panel { grid-column: 2; grid-row: 2; background: #fff; border-radius: 12px;
                          border: 1px solid #e0e0e0; display: flex; flex-direction: column; }
        .workflow-content { padding: 16px; flex: 1; overflow-y: auto; }

        /* Compact Workflow Steps */
        .steps-container { display: flex; flex-direction: column; gap: 8px; }
        .step { padding: 10px 12px; border-radius: 8px; background: #fafafa; border-left: 3px solid #ddd; transition: all 0.2s; }
        .step.active { border-left-color: #000; background: #f0f0f0; }
        .step.complete { border-left-color: #000; }
        .step.error { border-left-color: #c00; background: #fff5f5; }
        .step-header { display: flex; align-items: center; gap: 8px; }
        .step-num { width: 20px; height: 20px; border-radius: 50%; background: #ddd; display: flex;
                    align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: #666; }
        .step.complete .step-num { background: #000; color: #fff; }
        .step.error .step-num { background: #c00; color: #fff; }
        .step-title { font-size: 12px; font-weight: 600; color: #333; }
        .step-detail { font-size: 11px; color: #888; margin-top: 4px; margin-left: 28px; }
        /* Router Signal/Decision Steps - always larger and bold */
        .step.router-step .step-title { font-size: 14px; font-weight: 700; }
        .step.router-step .step-detail { font-size: 13px; font-weight: 600; }
        /* Router Steps - highlighted when complete */
        .step.router-step.complete { background: #f0f7ff; border-left-color: #0066cc; }
        .step.router-step.complete .step-num { background: #0066cc; color: #fff; }
        .step.router-step.complete .step-title { color: #0066cc; }
        .step.router-step.complete .step-detail { color: #0066cc; }

        /* Context Box - Full display */
        .context-box { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px;
                       margin-top: 8px; font-size: 12px; color: #444; line-height: 1.5; max-height: 150px;
                       overflow-y: auto; font-family: -apple-system, sans-serif; }
        .context-box:empty { display: none; }
        .context-label { font-size: 10px; color: #888; text-transform: uppercase; margin-bottom: 4px; }

        /* Quick Questions */
        .quick-section { padding: 16px 20px; border-top: 1px solid #e0e0e0; }
        .quick-section h3 { font-size: 11px; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .quick-btn { background: #fff; border: 1px solid #ddd; color: #333; padding: 8px 14px;
                     border-radius: 20px; cursor: pointer; margin: 3px; font-size: 12px; transition: all 0.2s; }
        .quick-btn:hover { border-color: #000; background: #f5f5f5; }

        /* Logo */
        .logo { height: 32px; vertical-align: middle; margin-right: 12px; }
        .header-row { display: flex; align-items: center; justify-content: center; gap: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-row">
                <img src="/vllm-logo.png" alt="vLLM" class="logo">
                <h1>Hallucination Gatekeeper</h1>
            </div>
            <p class="subtitle">Building Trustable AI with vLLM Semantic Router</p>
        </div>

        <div class="main-layout">
            <!-- Detection Result - Most Prominent -->
            <div class="detection-card" id="detectionCard">
                <div class="detection-title">Detection Result</div>
                <div class="detection-status" id="detectionStatus">
                    <span class="icon">◯</span>
                    <span>Waiting for input...</span>
                </div>
                <div class="detection-detail" id="detectionDetail">Ask a question to test hallucination detection</div>
                <div class="flagged-items" id="flaggedItems"></div>
            </div>

            <!-- Chat Panel -->
            <div class="chat-panel">
                <div class="panel-header">
                    <span>Chat</span>
                    <button class="new-chat-btn" onclick="newChat()" title="New Chat">+</button>
                </div>
                <div class="chat-messages" id="chatMessages"></div>
                <div class="chat-input-area">
                    <input type="text" id="chatInput" placeholder="Ask a question..." onkeypress="if(event.key==='Enter')sendMessage()">
                    <button onclick="sendMessage()" id="sendBtn">Send</button>
                    <label class="websearch-toggle">
                        <input type="checkbox" id="webSearchToggle" checked>
                        <span>🔍 WebSearch</span>
                    </label>
                </div>
                <div class="quick-section">
                    <h3>Quick Test</h3>
                    <button class="quick-btn" onclick="askQuestion('When was the Eiffel Tower built?')">🗼 Eiffel Tower (Factual)</button>
                    <button class="quick-btn" onclick="askQuestion('Who founded Apple Inc?')">🍎 Apple Founders (Factual)</button>
                    <button class="quick-btn" onclick="askQuestion('Write a short poem about technology')">✨ Write a Poem (Creative)</button>
                </div>
            </div>

            <!-- Workflow Panel -->
            <div class="workflow-panel">
                <div class="panel-header">Pipeline Status</div>
                <div class="workflow-content">
                    <div class="steps-container">
                        <div class="step" id="step1"><div class="step-header"><span class="step-num">1</span><span class="step-title">Question Sent</span></div><div class="step-detail" id="step1-detail">—</div></div>
                        <div class="step router-step" id="step2"><div class="step-header"><span class="step-num">2</span><span class="step-title">Router Signals:</span></div><div class="step-detail" id="step2-detail">—</div></div>
                        <div class="step" id="step3"><div class="step-header"><span class="step-num">3</span><span class="step-title">Tool Calls</span></div><div class="step-detail" id="step3-detail">—</div></div>
                        <div class="step" id="step4">
                            <div class="step-header"><span class="step-num">4</span><span class="step-title">Context Retrieved</span></div>
                            <div class="context-box" id="contextBox"></div>
                        </div>
                        <div class="step router-step" id="step5"><div class="step-header"><span class="step-num">5</span><span class="step-title">Router Decisions:</span></div><div class="step-detail" id="step5-detail">—</div></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const ROUTER_URL = '{{ROUTER_URL}}';
        const SEARCH_URL = '{{SEARCH_URL}}';

        function addMessage(role, content, typeEffect = false) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerHTML = '<div class="message-label">' + role + '</div><div class="bubble"></div>';
            document.getElementById('chatMessages').appendChild(div);
            const bubble = div.querySelector('.bubble');

            if (typeEffect && role === 'assistant') {
                // Typing effect for assistant messages
                let i = 0;
                const speed = 15; // ms per character
                function typeChar() {
                    if (i < content.length) {
                        bubble.textContent += content.charAt(i);
                        i++;
                        document.getElementById('chatMessages').scrollTop = 999999;
                        setTimeout(typeChar, speed);
                    }
                }
                typeChar();
            } else {
                bubble.textContent = content;
            }
            document.getElementById('chatMessages').scrollTop = 999999;
        }

        function updateStep(num, status, detail) {
            const step = document.getElementById('step' + num);
            if (!step) return;
            step.className = 'step ' + status;
            const detailEl = document.getElementById('step' + num + '-detail');
            if (detail && detailEl) detailEl.textContent = detail;
        }

        function resetSteps() {
            for (let i = 1; i <= 5; i++) { updateStep(i, '', '—'); }
            updateDetection('waiting', 'Waiting for input...', 'Ask a question to test hallucination detection', []);
        }

        function updateDetection(state, status, detail, flagged) {
            const card = document.getElementById('detectionCard');
            const statusEl = document.getElementById('detectionStatus');
            const detailEl = document.getElementById('detectionDetail');
            const flaggedEl = document.getElementById('flaggedItems');

            card.className = 'detection-card ' + state;

            if (state === 'detected') {
                statusEl.innerHTML = '<span class="icon">✕</span><span>Hallucination Detected</span>';
            } else if (state === 'safe') {
                statusEl.innerHTML = '<span class="icon">✓</span><span>Response Verified</span>';
            } else if (state === 'warning') {
                statusEl.innerHTML = '<span class="icon">⚠</span><span>Unverified Response</span>';
            } else if (state === 'checking') {
                statusEl.innerHTML = '<span class="icon">◐</span><span>Analyzing...</span>';
            } else {
                statusEl.innerHTML = '<span class="icon">◯</span><span>Waiting for input...</span>';
            }

            detailEl.textContent = detail;
            flaggedEl.innerHTML = flagged.map(f => '<span class="flagged-item">' + f + '</span>').join('');
        }

        function askQuestion(q) { document.getElementById('chatInput').value = q; sendMessage(); }

        function newChat() {
            document.getElementById('chatMessages').innerHTML = '';
            document.getElementById('chatInput').value = '';
            resetSteps();
            updateContext('');
        }

        function updateContext(text) {
            document.getElementById('contextBox').textContent = text || '';
        }

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const question = input.value.trim();
            if (!question) return;

            const enableWebSearch = document.getElementById('webSearchToggle').checked;

            input.value = '';
            document.getElementById('sendBtn').disabled = true;
            addMessage('user', question);
            resetSteps();
            updateContext('');
            updateDetection('checking', 'Analyzing...', 'Processing your question through the pipeline', []);

            try {
                updateStep(1, 'active', 'Sending...');
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question, enable_web_search: enableWebSearch})
                });
                const data = await resp.json();

                // Step 1: Question sent
                updateStep(1, 'complete', question.substring(0, 40) + (question.length > 40 ? '...' : ''));

                // Step 2: Router Signals - Need Fact Check
                const factCheckNeeded = data.headers && data.headers['x-vsr-fact-check-needed'] === 'true';
                updateStep(2, 'complete', factCheckNeeded ? 'Need Fact Check' : 'No Fact Check');

                // Check for unverified factual response (no web search)
                const unverifiedResponse = data.headers && data.headers['x-vsr-unverified-factual-response'] === 'true';
                const hallucinationDetected = data.headers && data.headers['x-vsr-hallucination-detected'] === 'true';

                // Step 3 & 4: Tool calls and context
                if (data.tool_calls) {
                    updateStep(3, 'complete', 'web_search called');
                    updateStep(4, 'complete', '');
                    updateContext(data.search_context || 'No context returned');
                } else {
                    updateStep(3, 'complete', enableWebSearch ? 'No tools needed' : 'WebSearch disabled');
                    updateStep(4, 'complete', '');
                    updateContext(unverifiedResponse ? 'No context (WebSearch disabled)' : '—');
                }

                // Step 5: Router Decisions - Hallucination Detection result
                if (unverifiedResponse) {
                    updateStep(5, 'complete', 'Unverified (no context)');
                } else if (hallucinationDetected) {
                    updateStep(5, 'complete', 'Hallucination Detected!');
                } else {
                    updateStep(5, 'complete', 'Response Verified');
                }

                // Detection result card
                const spans = data.headers ? (data.headers['x-vsr-hallucination-spans'] || '') : '';
                const flaggedList = spans ? spans.split(',').map(s => s.trim()).filter(s => s) : [];

                if (unverifiedResponse) {
                    updateDetection('warning', 'Unverified Response',
                        'Response contains factual claims that could not be verified (no fact-check context)', ['No verification context available']);
                } else if (hallucinationDetected) {
                    updateDetection('detected', 'Hallucination Detected',
                        'Response contains claims that conflict with retrieved context', flaggedList);
                } else {
                    updateDetection('safe', 'Response Verified',
                        'No hallucinations detected in the response', []);
                }

                addMessage(data.error ? 'error' : 'assistant', data.response || data.error, true);

            } catch (e) {
                addMessage('error', 'Error: ' + e.message, false);
                updateDetection('waiting', 'Error', e.message, []);
            }
            document.getElementById('sendBtn').disabled = false;
        }
    </script>
</body>
</html>"""


class ChatHandler(BaseHTTPRequestHandler):
    router_url = "http://localhost:8801"
    search_url = "http://localhost:8003"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    def log_message(self, format, *args):
        print(f"[WebClient] {args[0]}")

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = HTML_TEMPLATE.replace("{{ROUTER_URL}}", self.router_url).replace(
                "{{SEARCH_URL}}", self.search_url
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/vllm-logo.png":
            import os

            logo_path = os.path.join(os.path.dirname(__file__), "vllm-logo.png")
            if os.path.exists(logo_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(logo_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json({"error": "Not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        question = body.get("question", "")
        enable_web_search = body.get("enable_web_search", True)

        result = {
            "question": question,
            "tool_calls": None,
            "search_context": None,
            "response": None,
            "headers": {},
        }

        try:
            messages = [{"role": "user", "content": question}]

            if enable_web_search:
                # Mode 1: With WebSearch - Send to router with tools for hallucination detection
                resp_data, headers = self._send_to_router(messages, include_tools=True)

                if not resp_data:
                    result["error"] = "Failed to get response from router"
                    self.send_json(result)
                    return

                msg = resp_data.get("choices", [{}])[0].get("message", {})
                tool_calls = msg.get("tool_calls", [])

                if tool_calls:
                    result["tool_calls"] = tool_calls
                    messages.append(msg)

                    # Execute tool calls
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        args = json.loads(func.get("arguments", "{}"))

                        if func.get("name") == "web_search":
                            context = self._web_search(args.get("query", ""))
                            result["search_context"] = context
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", ""),
                                    "content": context,
                                }
                            )

                    # Get final response with tool results
                    resp_data, headers = self._send_to_router(
                        messages, include_tools=False
                    )
                    if resp_data:
                        msg = resp_data.get("choices", [{}])[0].get("message", {})

                result["response"] = msg.get("content", "")
                result["headers"] = headers
            else:
                # Mode 2: Without WebSearch - Direct response, will get unverified factual response header
                resp_data, headers = self._send_to_router(messages, include_tools=False)

                if not resp_data:
                    result["error"] = "Failed to get response from router"
                    self.send_json(result)
                    return

                msg = resp_data.get("choices", [{}])[0].get("message", {})
                result["response"] = msg.get("content", "")
                result["headers"] = headers

        except Exception as e:
            result["error"] = str(e)

        self.send_json(result)

    def _send_to_router(self, messages, include_tools):
        payload = {"model": "qwen3", "messages": messages, "max_tokens": 512}
        if include_tools:
            payload["tools"] = self.tools

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.router_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if "hallucination" in k.lower()
                    or "fact-check" in k.lower()
                    or "unverified" in k.lower()
                }
                return json.loads(resp.read().decode()), headers
        except Exception as e:
            print(f"[WebClient] Router error: {e}")
            return None, {}

    def _web_search(self, query):
        try:
            data = json.dumps({"query": query}).encode()
            req = urllib.request.Request(
                f"{self.search_url}/search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode()).get("context", "No results")
        except Exception as e:
            return f"Search error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Web-based Hallucination Demo Client")
    parser.add_argument("--port", type=int, default=8888, help="Web server port")
    parser.add_argument("--router-url", type=str, default="http://localhost:8801")
    parser.add_argument("--search-url", type=str, default="http://localhost:8003")
    args = parser.parse_args()

    ChatHandler.router_url = args.router_url
    ChatHandler.search_url = args.search_url

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    print(f"\n🌐 Hallucination Detection Demo")
    print(f"   Open in browser: http://localhost:{args.port}")
    print(f"   Router: {args.router_url}")
    print(f"   Search: {args.search_url}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
