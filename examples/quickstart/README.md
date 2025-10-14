# Semantic Router Quickstart

This quickstart walks through the minimal set of commands needed to prove that
the semantic router can classify incoming chat requests, route them through
Envoy, and receive OpenAI-compatible completions. The flow is optimized for
local laptops and uses a lightweight mock backend by default, so the entire
loop finishes in a few minutes.

## Prerequisites

- Python environment with the project’s dependencies and virtualenv activated.
- `make`, `curl`, `go`, `cargo`, `rustc`, and `python3` in `PATH`.
- All commands below are run from the repository root.

## Step-by-Step Runbook

0. **Download router support models**

   These assets (ModernBERT classifiers, LoRA adapters, embeddings, etc.) are
   required before the router can start.

   ```bash
   make download-models
   ```

1. **Start the OpenAI-compatible backend**

   The router expects at least one endpoint that serves `/v1/chat/completions`.
   You can point to a real vLLM deployment, but the fastest option is the
   bundled mock server:

   ```bash
   pip install -r tools/mock-vllm/requirements.txt
   python -m uvicorn tools.mock_vllm.app:app --host 0.0.0.0 --port 8000
   ```

   Leave this process running; it provides instant canned responses for
   `openai/gpt-oss-20b`.

2. **Launch Envoy**

   In a separate terminal, bring up the Envoy sidecar that listens on
   `http://127.0.0.1:8801/v1/*` and forwards traffic to the router’s gRPC
   ExtProc server.

   ```bash
   make run-envoy
   ```

3. **Start the router with the quickstart config**

   In another terminal, run the quickstart bootstrap. Point the health probe at
   the router’s local HTTP API (port 8080) so the script does not wait on the
   Envoy endpoint.

   ```bash
   QUICKSTART_HEALTH_URL=http://127.0.0.1:8080/health \
     ./examples/quickstart/quickstart.sh --skip-download --skip-build
   ```

   Keep this process alive; Ctrl+C will stop the router.

4. **Run the quick evaluation**

   With Envoy, the router, and the mock backend running, execute the benchmark
   to send a small batch of MMLU questions through the routing pipeline.

   ```bash
   OPENAI_API_KEY="sk-test" \
     ./examples/quickstart/quick-eval.sh \
       --mode router \
       --samples 5 \
       --vllm-endpoint ""
   ```

   - `--mode router` restricts the run to router-transparent requests.
   - `--vllm-endpoint ""` disables direct vLLM comparisons.

5. **Inspect the results**

   The evaluator writes all artifacts under
   `examples/quickstart/results/<timestamp>/`:

   - `raw/` – individual JSON summaries per dataset/model combination.
   - `quickstart-summary.csv` – tabular metrics (accuracy, tokens, latency).
   - `quickstart-report.md` – Markdown report suitable for sharing.

   You can re-run the evaluator with different flags (e.g., `--samples 10`,
   `--dataset arc`) and the outputs will land in fresh timestamped folders.

## Switching to a Real vLLM Backend

If you prefer to exercise a real language model:

1. Replace step 1 with a real vLLM launch (or any OpenAI-compatible server).
2. Update `examples/quickstart/config-quickstart.yaml` so the `vllm_endpoints`
   block points to that service (IP, port, and model name).
3. Re-run steps 2–4. No other changes to the quickstart scripts are needed.

Keep the mock server documented for quick demos; swap to full vLLM when you
want latency/quality signals from the actual model.
