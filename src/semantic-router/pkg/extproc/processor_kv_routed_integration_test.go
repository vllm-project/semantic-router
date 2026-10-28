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
	live := sourceURL != "" && targetURL != ""
	if (sourceURL == "") != (targetURL == "") || (os.Getenv("KV_TEST_MAPPER_ID") != "" && !live) || live && os.Getenv("KV_TEST_MAPPER_ID") == "" {
		t.Fatal("live validation requires both server URLs and the mapper ID")
	}
	if !live {
		backend := func(hint *map[string]string) *httptest.Server {
			return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					KV map[string]string `json:"kv_transfer_params"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				*hint = body.KV
				if body.KV["mapper_id"] != "" && expectedStatus != "" {
					w.Header().Set("x-vsr-kv-transfer-status", expectedStatus)
					reason := "loaded"
					if expectedStatus == "fallback_reprefill" {
						reason = "snapshot_missing"
					}
					w.Header().Set("x-vsr-kv-transfer-reason", reason)
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
	call := func(principal string, history []map[string]any) (http.Header, map[string]any) {
		t.Helper()
		data, _ := json.Marshal(map[string]any{"model": "vllm-sr/auto", "messages": history, "max_tokens": 1, "temperature": 0})
		req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(string(data)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-authz-user-id", principal)
		req.Header.Set("x-session-id", "routed-test")
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
	_, response := call("tenant-a", messages)
	record, err := registry.Lookup(context.Background(), cache.UserScopeNamespace("tenant-a"), sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "routed-test"))
	if err != nil || record == nil || record.CacheID == "" {
		t.Fatalf("source publication: %v, %v", record, err)
	}
	assistant := response["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
	if assistant == nil {
		assistant = ""
	}
	messages = append(messages, map[string]any{"role": "assistant", "content": assistant}, map[string]any{"role": "user", "content": "Please provide an advanced answer."})
	result, targetResponse := call("tenant-a", messages)
	if result.Get("x-vsr-kv-transfer-status") != expectedStatus {
		t.Fatalf("target outcome: got %v, want %q", result, expectedStatus)
	}
	if !live && (sourceHint["mapper_id"] != "" || targetHint["cache_id"] != record.CacheID || targetHint["mapper_id"] != mapper) {
		t.Fatalf("wrong hints: %v / %v", sourceHint, targetHint)
	}
	result, coldResponse := call("tenant-b", messages)
	if result.Get("x-vsr-kv-transfer-status") != "" {
		t.Fatal("another tenant attempted reuse")
	}
	if !live && len(targetHint) != 0 {
		t.Fatal("another tenant received a transfer hint")
	}
	if expectedStatus != "applied" && !reflect.DeepEqual(targetResponse["choices"], coldResponse["choices"]) {
		t.Fatal("stock or fallback output differed from normal prefill")
	}
}
