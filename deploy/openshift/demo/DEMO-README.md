# Demo Scripts for Semantic Router

This directory contains demo scripts to showcase the semantic router capabilities.

## Quick Demo Guide

### 1. Live Log Viewer (Run in Terminal 1)

Shows real-time classification, routing, and security decisions:

```bash
./deploy/openshift/demo/live-semantic-router-logs.sh
```

**What it shows:**

- 📨 **Incoming requests** with user prompts
- 🛡️ **Security checks** (jailbreak detection)
- 🔍 **Classification** (category detection with confidence)
- 🎯 **Routing decisions** (which model was selected)
- 💾 **Cache hits** (semantic similarity matching)
- 🧠 **Reasoning mode** activation

**Tip:** Run this in a split terminal or separate window during your demo!

---

### 2. Interactive Demo (Run in Terminal 2)

Interactive menu-driven semantic router demo:

```bash
python3 deploy/openshift/demo/demo-semantic-router.py
```

**Features:**

1. **Single Classification** - Tests random prompt from golden set
2. **All Classifications** - Tests all 10 golden prompts
3. **PII Detection Test** - Tests personal information filtering
4. **Jailbreak Detection Test** - Tests security filtering
5. **Run All Tests** - Executes all tests sequentially

**Requirements:**

- ✅ Must be logged into OpenShift (`oc login`)
- URLs are discovered automatically from routes

**What it does:**

- Goes through Envoy (same path as OpenWebUI)
- Shows routing decisions and response previews
- **Appears in Grafana dashboard!**
- Interactive - choose what to test

---

## Demo Flow Suggestion

### Setup (Before Demo)

```bash
# Terminal 1: Start log viewer
./deploy/openshift/demo/live-semantic-router-logs.sh

# Terminal 2: Ready to run classification test
# (don't run yet)

# Browser Tab 1: Open Grafana
# http://grafana-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com

# Browser Tab 2: Open OpenWebUI
# http://openwebui-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com
```

### During Demo

1. **Show the system overview**
   - Explain semantic routing concept
   - Show the architecture diagram

2. **Run interactive demo** (Terminal 2)

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   Choose option 2 (All Classifications)

3. **Point to live logs** (Terminal 1)
   - Show real-time classification
   - Highlight security checks (jailbreak: BENIGN)
   - Show routing decisions (Model-A vs Model-B)
   - Point out cache hits

4. **Switch to Grafana** (Browser Tab 1)
   - Show request metrics appearing
   - Show classification category distribution
   - Show model usage breakdown

5. **Show OpenWebUI integration** (Browser Tab 2)
   - Type one of the golden prompts
   - Watch it appear in logs (Terminal 1)
   - Show the same routing happening

---

## Key Talking Points

### Classification Accuracy

- **10 golden prompts** with 100% accuracy
- Categories: Chemistry, History, Psychology, Health, Math
- Shows consistent classification behavior

### Security Features

- **Jailbreak detection** on every request
- Shows "BENIGN" for safe requests
- Confidence scores displayed

### Smart Routing

- Automatic model selection based on content
- Load balancing across Model-A and Model-B
- Routing decisions visible in logs

### Performance

- **Semantic caching** reduces latency
- Cache hits shown in logs with similarity scores
- Sub-second response times

### Observability

- Real-time logs with structured JSON
- Grafana metrics and dashboards
- Request tracing and debugging

---

## Troubleshooting

### Log viewer shows no output

```bash
# Check if semantic-router pod is running
oc get pods -n vllm-semantic-router-system | grep semantic-router

# Check logs manually
oc logs -n vllm-semantic-router-system deployment/semantic-router --tail=20
```

### Classification test fails

```bash
# Verify Envoy route is accessible
curl http://envoy-http-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com/v1/models

# Check if models are ready
oc get pods -n vllm-semantic-router-system
```

### Grafana doesn't show metrics

- Wait 15-30 seconds for metrics to appear
- Refresh the dashboard
- Check the time range (last 5 minutes)

---

## Cache Management

### Check Cache Status

```bash
./deploy/openshift/demo/cache-management.sh status
```

Shows recent cache activity and cached queries.

### Clear Cache (for demo)

```bash
./deploy/openshift/demo/cache-management.sh clear
```

Restarts semantic-router deployment to clear in-memory cache (~30 seconds).

### Demo Cache Feature

**Workflow to show caching in action:**

1. Clear the cache:

   ```bash
   ./deploy/openshift/demo/cache-management.sh clear
   ```

2. Run classification test (first time - no cache):

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   Choose option 2 (All Classifications)
   - Processing time: ~3-4 seconds per query
   - Logs show queries going to model

3. Run classification test again (second time - with cache):

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   Choose option 2 (All Classifications) again
   - Processing time: ~400ms per query (10x faster!)
   - Logs show "💾 CACHE HIT" for all queries
   - Similarity scores ~0.99999

**Key talking point:** Cache uses **semantic similarity**, not exact string matching!

---

## Files

- `live-semantic-router-logs.sh` - Envoy traffic log viewer (security, cache, routing)
- `live-classifier-logs.sh` - Classification API log viewer
- `demo-semantic-router.py` - Interactive demo with multiple test options
- `curl-examples.sh` - Quick classification examples (direct API)
- `cache-management.sh` - Cache management helper
- `CATEGORY-MODEL-MAPPING.md` - Category to model routing reference
- `demo-classification-results.json` - Test results (auto-generated)

---

## Notes

- The log viewer uses `oc logs --follow`, so it will run indefinitely until you press Ctrl+C
- Classification test takes ~60 seconds (10 prompts with 0.5s delay between each)
- All requests go through Envoy, triggering the full routing pipeline
- Grafana metrics update in real-time (with slight delay)
