# Computer Use Agent Service

This service provides an E2B-powered desktop automation agent for the vLLM Semantic Router Dashboard. It enables vision-language models to control a remote desktop environment through a web interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Computer Use Agent Page (React)                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │   VNC Viewer        │  │   Steps Timeline                 │  │
│  │   (Live Desktop)    │  │   [1] Open Google                │  │
│  │   ┌───────────────┐ │  │   [2] Click search bar           │  │
│  │   │  E2B Desktop  │ │  │   [3] Type query                 │  │
│  │   │  via iframe   │ │  │   [4] Press Enter                │  │
│  │   └───────────────┘ │  │   [5] Read results               │  │
│  └─────────────────────┘  └─────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
         │                                           │
         │ WebSocket /ws                             │ REST API
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python Agent Service (FastAPI - Port 8000)          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ SandboxService   │  │ AgentService     │  │ WebSocketMgr   │ │
│  │ - E2B pool mgmt  │  │ - Agent loop     │  │ - Real-time    │ │
│  │ - VNC URLs       │  │ - Step tracking  │  │ - Progress     │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         │ E2B SDK
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    E2B Cloud (Firecracker microVMs)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Desktop Sandbox (Ubuntu + GUI + Firefox + Terminal)      │  │
│  │  - screenshot() → Image bytes                             │  │
│  │  - mouse.left_click(x, y)                                 │  │
│  │  - keyboard.write(text)                                   │  │
│  │  - keyboard.press(key)                                    │  │
│  │  - get_vnc_url() → Stream URL                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

1. **Install dependencies**:
   ```bash
   cd dashboard/agent-service
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the service**:
   ```bash
   python -m cua_agent.main
   ```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `E2B_API_KEY` | E2B Cloud API key | Yes |
| `HF_TOKEN` | HuggingFace token for Qwen models | One of HF/Ollama/OpenAI |
| `OLLAMA_HOST` | Ollama server URL (local models) | One of HF/Ollama/OpenAI |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4V | One of HF/Ollama/OpenAI |
| `HOST` | Server bind address | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8000) |
| `MAX_SANDBOXES` | Max concurrent sandboxes | No (default: 10) |

## Available Tools

The agent has access to 12 desktop automation tools:

| Tool | Description |
|------|-------------|
| `click(x, y)` | Left-click at coordinates (0-1000 normalized) |
| `right_click(x, y)` | Right-click |
| `double_click(x, y)` | Double-click |
| `move_mouse(x, y)` | Move cursor |
| `write(text)` | Type text |
| `press(keys)` | Press keyboard keys |
| `scroll(x, y, direction, amount)` | Scroll page |
| `wait(seconds)` | Wait N seconds |
| `open_url(url)` | Open URL in browser |
| `launch(app)` | Launch application |
| `go_back()` | Browser back button |
| `drag(x1, y1, x2, y2)` | Drag and drop |

## API Endpoints

### REST API

- `GET /health` - Health check
- `GET /api/health` - Detailed health with WebSocket connections
- `GET /api/models` - List available vision models

### WebSocket

- `WS /ws` - Real-time agent communication

**Messages:**
- `user_task` - Submit task with AgentTrace
- `stop_task` - Stop running task

**Events:**
- `heartbeat` - Welcome message with UUID
- `agent_start` - Task started
- `agent_progress` - Step completed with screenshot
- `agent_complete` - Task finished
- `vnc_url_set` - VNC stream URL
- `vnc_url_unset` - Clear VNC stream
- `agent_error` - Error occurred

## Supported Models

### HuggingFace Inference API
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-235B-A22B-Instruct`

### Ollama (Local)
- `ollama/qwen2-vl:7b`
- `ollama/llava:7b`
- `ollama/llava:13b`

### OpenAI
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

## Using Makefile Targets

From the project root, you can use these Makefile targets:

```bash
# Install agent service dependencies
make dashboard-agent-install

# Start agent service in dev mode
make dashboard-agent-dev

# See all available dashboard targets
make dashboard-dev-all
```

## Integration with Dashboard

The agent service integrates with the Go backend via WebSocket and REST proxying:

1. Start the agent service (port 8000)
2. Start the Go backend with agent URL:
   ```bash
   cd dashboard/backend
   go run main.go -agent=http://localhost:8000
   ```
3. Start the frontend
4. Navigate to the "E2B Desktop Agent" tab

The Go backend proxies:
- `/ws/agent` → Agent WebSocket
- `/api/agent/*` → Agent REST endpoints
