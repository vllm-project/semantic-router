# Multi-Instance Demo Script

## Terminal Commands to Record

### Terminal 1: Start first instance (gpt-3.5-turbo)

```bash
# Clear screen
clear

# Install (simulate - already installed)
echo "$ pip install llm-katan"
echo "Requirement already satisfied: llm-katan"

# Start first server
echo "$ llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name 'gpt-3.5-turbo'"
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name "gpt-3.5-turbo" &
sleep 3
```

### Terminal 2: Start second instance (claude-3-haiku)

```bash
clear
echo "$ llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name 'claude-3-haiku'"
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "claude-3-haiku" &
sleep 3
```

### Terminal 3: Test both endpoints

```bash
clear
echo "$ curl http://localhost:8000/v1/models | jq '.data[0].id'"
curl -s http://localhost:8000/v1/models | jq '.data[0].id'

echo ""
echo "$ curl http://localhost:8001/v1/models | jq '.data[0].id'"
curl -s http://localhost:8001/v1/models | jq '.data[0].id'

echo ""
echo "# Same tiny model, different API names for testing!"
```

## Key Points to Highlight

- One tiny model (Qwen3-0.6B)
- Two different API endpoints
- Different model names served
- Perfect for testing multi-provider scenarios
- Minimal resource usage
