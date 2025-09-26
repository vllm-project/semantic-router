# Semantic Router Test Suite

This test suite provides a progressive approach to testing the Semantic Router, following the data flow from client request to final response.

## Test Flow

1. **00-client-request-test.py** - Basic client request tests ✅
   - Tests sending requests to the Envoy proxy
   - Verifies basic request formatting and endpoint availability
   - Tests malformed request validation
   - Tests content-based smart routing (math → TinyLlama, creative → Qwen)

2. **01-envoy-extproc-test.py** - TBD (To Be Developed)
   - Tests that Envoy correctly forwards requests to the ExtProc
   - Checks header propagation

3. **02-router-classification-test.py** - TBD (To Be Developed)
   - Tests BERT embeddings
   - Tests category classification
   - Verifies model selection based on content

4. **03-model-routing-test.py** - TBD (To Be Developed)
   - Tests that requests are routed to the correct backend model
   - Verifies model header modifications

5. **04-cache-test.py** - TBD (To Be Developed)
   - Tests cache hit/miss behavior
   - Verifies similarity thresholds
   - Tests cache TTL

6. **05-e2e-category-test.py** - TBD (To Be Developed)
   - Tests math queries route to the math-specialized model
   - Tests creative queries route to the creative-specialized model
   - Tests other domain-specific routing

7. **06-metrics-test.py** - TBD (To Be Developed)
   - Tests Prometheus metrics endpoints
   - Verifies correct metrics are being recorded

## Running Tests

### Development Workflow (LLM Katan - Recommended)

For fast development and testing with real tiny models (no GPU required):

```bash
# Terminal 1: Start LLM Katan servers (shows request logs, Ctrl+C to stop)
./e2e-tests/start-llm-katan.sh

# Or manually start individual servers:
llm-katan --model Qwen/Qwen3-0.6B --port 8000 --served-model-name "Qwen/Qwen2-0.5B-Instruct"
llm-katan --model Qwen/Qwen3-0.6B --port 8001 --served-model-name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Terminal 2: Start Envoy proxy
make run-envoy

# Terminal 3: Start semantic router
make run-router

# Terminal 4: Run tests
python e2e-tests/00-client-request-test.py    # Individual test
python e2e-tests/run_all_tests.py             # All available tests
```

**Note**: The LLM Katan servers use real tiny models for actual inference while being lightweight enough for development. The script runs in foreground mode, allowing you to see real-time request logs and use Ctrl+C to stop all servers cleanly.

### Future: Production Testing (Real vLLM)

Will be added in future PRs for testing with actual model inference.

## Available Tests

Currently implemented:

- **00-client-request-test.py** ✅ - Complete client request validation and smart routing

Individual tests can be run with:

```bash
python e2e-tests/00-client-request-test.py
```

Or run all available tests with:

```bash
python e2e-tests/run_all_tests.py
```
