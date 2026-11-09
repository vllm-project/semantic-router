package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// Set KV_TEST_SOURCE_URL, KV_TEST_TARGET_URL and KV_TEST_MAPPER_ID to run against live vLLM.
func TestKVRoutedSourceAndTarget(t *testing.T) {
	if os.Getenv("KV_TEST_SOURCE_URL") != "" || os.Getenv("KV_TEST_TARGET_URL") != "" || os.Getenv("KV_TEST_MAPPER_ID") != "" {
		status := "applied"
		if os.Getenv("KV_TEST_STOCK_BACKEND") == "true" {
			status = ""
		}
		testKVRoutedSourceAndTarget(t, status)
		return
	}
	for _, status := range []string{"applied", "fallback_reprefill", ""} {
		name := status
		if name == "" {
			name = "stock_backend"
		}
		t.Run(name, func(t *testing.T) { testKVRoutedSourceAndTarget(t, status) })
	}
}

func testKVRoutedSourceAndTarget(t *testing.T, expectedStatus string) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-only-secret")
	sourceURL, targetURL := os.Getenv("KV_TEST_SOURCE_URL"), os.Getenv("KV_TEST_TARGET_URL")
	var sourceHint, targetHint map[string]string
	var hintMu sync.Mutex
	live := sourceURL != "" && targetURL != ""
	if (sourceURL == "") != (targetURL == "") || (os.Getenv("KV_TEST_MAPPER_ID") != "" && !live) || live && os.Getenv("KV_TEST_MAPPER_ID") == "" {
		t.Fatal("live validation requires both server URLs and the mapper ID")
	}
	if !live {
		backend := func(hint *map[string]string) *httptest.Server {
			return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					KV     map[string]string `json:"kv_transfer_params"`
					Stream bool              `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				hintMu.Lock()
				*hint = body.KV
				hintMu.Unlock()
				if body.KV["mapper_id"] != "" && expectedStatus != "" {
					w.Header().Set("x-vsr-kv-transfer-status", expectedStatus)
					reason := "loaded"
					if expectedStatus == "fallback_reprefill" {
						reason = "snapshot_missing"
					}
					w.Header().Set("x-vsr-kv-transfer-reason", reason)
				}
				if body.Stream {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"test","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
			}))
		}
		source, target := backend(&sourceHint), backend(&targetHint)
		defer source.Close()
		defer target.Close()
		sourceURL, targetURL = source.URL, target.URL
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.3
providers:
  defaults: {model: source}
  models:
    - name: source
      provider_model_id: Qwen/Qwen3-14B
      api_format: openai
      backend_refs: [{name: source-backend, provider: vllm, base_url: %s/v1}]
    - name: target
      provider_model_id: Qwen/Qwen3-32B
      api_format: openai
      backend_refs: [{name: target-backend, provider: vllm, base_url: %s/v1}]
routing:
  signals:
    keywords: [{name: advanced, operator: OR, keywords: [advanced]}]
  decisions:
    - name: advanced
      rules: {type: keyword, name: advanced}
      modelRefs: [{model: target}]
`, sourceURL, targetURL)))
	if err != nil {
		t.Fatal(err)
	}
	source := config.KVServingIdentity{Model: "Qwen/Qwen3-14B", WeightRevision: "40c069824f4251a91eefaf281ebe4c544efd3e18", Tokenizer: "Qwen/Qwen3-14B", TokenizerRevision: "40c069824f4251a91eefaf281ebe4c544efd3e18", Precision: "bf16", TensorParallel: 1, KVHeads: 8, HeadDim: 128, HeadOrder: "contiguous"}
	target := source
	target.Model = "Qwen/Qwen3-32B"
	target.WeightRevision = "9216db5781bf21249d130ec9da846c4624c16137"
	mapper := os.Getenv("KV_TEST_MAPPER_ID")
	if mapper == "" {
		mapper = "fixture-mapper"
	}
	cfg.KVTransfer = &config.KVTransferConfig{
		Enabled:  true,
		Backends: []config.KVTransferBackend{{ModelAlias: "source", BackendName: cfg.ModelConfig["source"].PreferredEndpoints[0], Identity: source, CanExport: true}, {ModelAlias: "target", BackendName: cfg.ModelConfig["target"].PreferredEndpoints[0], Identity: target, CanLoad: true}},
		Pairs:    []config.KVTransferPair{{MapperID: mapper, Source: source, Target: target, Enabled: true, MaxTransferTurn: 3}},
	}
	router := newParityRouter(t, cfg)
	redis := miniredis.RunT(t)
	registry, err := kvtransfer.NewRedisAddressRegistry(kvtransfer.RedisAddressRegistryOptions{Address: redis.Addr()})
	if err != nil {
		t.Fatal(err)
	}
	router.kvAddressRegistryStore = registry
	t.Cleanup(func() { _ = registry.Close() })
	remote, err := upstream.Build(cfg, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close(context.Background()) })
	handler, err := gateway.NewHandler(gateway.Options{Serving: gateway.Static(gateway.Serving{Engine: routing.NewEngine(router, routing.DefaultOptions), Upstream: remote, TrustIdentity: gateway.IdentityTrust{Headers: true}})})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()
	text := os.Getenv("KV_TEST_SOURCE_TEXT")
	if text == "" {
		text = strings.Repeat("Hello ", 128)
	}
	messages := []map[string]any{{"role": "user", "content": text}}
	call := func(principal, session string, history []map[string]any) (http.Header, map[string]any) {
		t.Helper()
		data, _ := json.Marshal(map[string]any{"model": "vllm-sr/auto", "messages": history, "max_tokens": 1, "temperature": 0})
		req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(string(data)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-authz-user-id", principal)
		req.Header.Set("x-session-id", session)
		resp, requestErr := server.Client().Do(req)
		if requestErr != nil {
			t.Fatal(requestErr)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			t.Fatalf("HTTP %d: %s", resp.StatusCode, body)
		}
		var wire map[string]any
		if decodeErr := json.Unmarshal(body, &wire); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		t.Logf("principal=%s transfer=%s response=%s", principal, resp.Header.Get("x-vsr-kv-transfer-status"), body)
		return resp.Header, wire
	}
	_, response := call("tenant-a", "routed-test", messages)
	record, err := registry.Lookup(context.Background(), cache.UserScopeNamespace("tenant-a"), sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "routed-test"))
	if err != nil || record == nil || record.CacheID == "" {
		t.Fatalf("source publication: %v, %v", record, err)
	}
	assistant := response["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
	if assistant == nil {
		assistant = ""
	}
	messages = append(messages, map[string]any{"role": "assistant", "content": assistant}, map[string]any{"role": "user", "content": "Please provide an advanced answer."})
	result, targetResponse := call("tenant-a", "routed-test", messages)
	if result.Get("x-vsr-kv-transfer-status") != expectedStatus {
		t.Fatalf("target outcome: got %v, want %q", result, expectedStatus)
	}
	hintMu.Lock()
	sourceObserved, targetObserved := sourceHint, targetHint
	hintMu.Unlock()
	if !live && (sourceObserved["mapper_id"] != "" || targetObserved["cache_id"] != record.CacheID || targetObserved["mapper_id"] != mapper) {
		t.Fatalf("wrong hints: %v / %v", sourceObserved, targetObserved)
	}
	result, coldResponse := call("tenant-b", "routed-test", messages)
	if result.Get("x-vsr-kv-transfer-status") != "" {
		t.Fatal("another tenant attempted reuse")
	}
	hintMu.Lock()
	targetObserved = targetHint
	hintMu.Unlock()
	if !live && len(targetObserved) != 0 {
		t.Fatal("another tenant received a transfer hint")
	}
	if expectedStatus != "applied" && !reflect.DeepEqual(targetResponse["choices"], coldResponse["choices"]) {
		t.Fatal("stock or fallback output differed from normal prefill")
	}

	// Streaming stays on normal prefill; the native API rejects caller hints.
	call("tenant-a", "stream-routing", messages[:1])
	streamRequest := map[string]any{
		"model": "vllm-sr/auto", "messages": messages, "max_tokens": 8, "temperature": 0, "stream": true,
	}
	for _, forged := range []bool{true, false} {
		if forged {
			streamRequest["kv_transfer_params"] = map[string]string{"namespace": "forged", "cache_id": "forged", "mapper_id": mapper}
		} else {
			delete(streamRequest, "kv_transfer_params")
		}
		streamData, _ := json.Marshal(streamRequest)
		streamReq, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(string(streamData)))
		streamReq.Header.Set("Content-Type", "application/json")
		streamReq.Header.Set("x-authz-user-id", "tenant-a")
		streamReq.Header.Set("x-session-id", "stream-routing")
		stream, streamErr := server.Client().Do(streamReq)
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		streamBody, readErr := io.ReadAll(stream.Body)
		_ = stream.Body.Close()
		if forged {
			if readErr != nil || stream.StatusCode != http.StatusBadRequest {
				t.Fatalf("caller hint was not rejected: %v, HTTP %d, %s", readErr, stream.StatusCode, streamBody)
			}
			continue
		}
		if readErr != nil || stream.StatusCode != http.StatusOK || !strings.Contains(string(streamBody), "data: [DONE]") {
			t.Fatalf("stream failed: %v, HTTP %d, %s", readErr, stream.StatusCode, streamBody)
		}
		if stream.Header.Get("x-vsr-kv-transfer-status") != "" || strings.Contains(string(streamBody), "vsr_kv_transfer") {
			t.Fatal("streaming request exposed a transfer outcome")
		}
	}
	hintMu.Lock()
	targetObserved = targetHint
	hintMu.Unlock()
	if !live && len(targetObserved) != 0 {
		t.Fatal("streaming request sent a caller transfer hint")
	}

	var calls sync.WaitGroup
	for i := range 4 {
		calls.Add(1)
		go func() {
			defer calls.Done()
			principal := fmt.Sprintf("parallel-tenant-%d", i)
			_, sourceResponse := call(principal, "parallel-routing", messages[:1])
			parallelRecord, lookupErr := registry.Lookup(context.Background(), cache.UserScopeNamespace(principal), sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "parallel-routing"))
			if lookupErr != nil || parallelRecord == nil || parallelRecord.CacheID == "" {
				t.Errorf("parallel source publication: %v, %v", parallelRecord, lookupErr)
				return
			}
			content := sourceResponse["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
			if content == nil {
				content = ""
			}
			history := []map[string]any{messages[0], {"role": "assistant", "content": content}, messages[2]}
			parallelResult, _ := call(principal, "parallel-routing", history)
			if parallelResult.Get("x-vsr-kv-transfer-status") != expectedStatus {
				t.Errorf("parallel target outcome: %v", parallelResult)
			}
		}()
	}
	calls.Wait()
}
