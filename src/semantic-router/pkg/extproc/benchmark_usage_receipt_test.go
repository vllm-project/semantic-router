package extproc

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestBenchmarkUsageDirectProofAndCallLimit(t *testing.T) {
	hash := strings.Repeat("a", 64)
	router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: hash}}
	for _, tc := range []struct {
		name, algorithm, plugin string
		cache                   bool
		known                   bool
	}{
		{name: "static", algorithm: "static", known: true},
		{name: "multi_factor", algorithm: "multi_factor", known: true},
		{name: "prompt helper", algorithm: "prompt"},
		{name: "looper", algorithm: "fusion"},
		{name: "shadow call", algorithm: "static", plugin: "shadow_dispatch"},
		{name: "memory", algorithm: "static", plugin: "memory"},
		{name: "unknown future plugin", algorithm: "static", plugin: "future"},
		{name: "cache hit", algorithm: "static", cache: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := &config.Decision{Algorithm: &config.AlgorithmConfig{Type: tc.algorithm}}
			if tc.plugin != "" {
				d.Plugins = []config.DecisionPlugin{{Type: tc.plugin}}
			}
			ctx := &RequestContext{Headers: map[string]string{headers.SRBenchExpectedConfigHash: hash, headers.SRBenchMaxInferenceCalls: "1"}, VSRSelectedDecision: d, VSRSelectedModel: "small", UpstreamStatusCode: 200, VSRCacheHit: tc.cache}
			mutation := &ext_proc.HeaderMutation{}
			response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ResponseHeaders{ResponseHeaders: &ext_proc.HeadersResponse{Response: &ext_proc.CommonResponse{HeaderMutation: mutation}}}}
			router.bindBenchmarkConfigResponse(response, ctx)
			count := ""
			for _, option := range mutation.SetHeaders {
				if option.Header.Key == headers.VSRInferenceCallCount {
					count = string(option.Header.RawValue)
				}
			}
			if (count == "1") != tc.known {
				t.Fatalf("unsafe call proof=%q", count)
			}
			if !tc.cache && (router.benchmarkCallLimitCheck(ctx) == nil) != tc.known {
				t.Fatal("call precondition mismatch")
			}
		})
	}
}

func TestBenchmarkUsageLooperIncludesEveryCallAndFailsClosed(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: "fusion"}}}
	var usage looper.TokenUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":30,"created_cache_tokens":20}}`), &usage); err != nil {
		t.Fatal(err)
	}
	resp := &looper.Response{Usage: usage.Add(&looper.ModelResponse{Usage: usage}), ExecutionTrace: looper.ExecutionTrace{Version: 1, Attempts: []looper.AttemptTrace{
		{Model: "a", Role: "candidate", Stage: "panel", Status: looper.AttemptStatusSucceeded, Usage: usage},
		{Model: "b", Role: "synthesis", Stage: "final", Status: looper.AttemptStatusSucceeded, Usage: usage},
	}}}
	read := func() benchmarkUsageReceipt {
		var receipt benchmarkUsageReceipt
		if err := json.Unmarshal([]byte(router.benchmarkLooperUsage(resp, ctx)), &receipt); err != nil {
			t.Fatal(err)
		}
		return receipt
	}
	receipt := read()
	if !receipt.Complete || len(receipt.Calls) != 2 || receipt.Calls[1].Usage.Write != 20 {
		t.Fatalf("incomplete receipt: %+v", receipt)
	}
	resp.ExecutionTrace.AttemptsTruncated = true
	if read().Complete {
		t.Fatal("truncated trace marked complete")
	}
	resp.ExecutionTrace.AttemptsTruncated = false
	resp.ExecutionTrace.Attempts[1].Usage.Unreported = true
	if read().Complete {
		t.Fatal("missing usage marked complete")
	}
	resp.ExecutionTrace.Attempts[1].Usage.Unreported = false
	if err := json.Unmarshal([]byte(`{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":30,"created_cache_tokens":20,"cache_creation_tokens":19}}`), &resp.ExecutionTrace.Attempts[1].Usage); err != nil {
		t.Fatal(err)
	}
	if read().Complete {
		t.Fatal("conflicting cache-write aliases marked complete")
	}
	resp.ExecutionTrace.Attempts[1].Usage = usage
	ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{Type: "shadow_dispatch"}}
	if read().Complete {
		t.Fatal("untraced shadow calls omitted")
	}
	ctx.VSRSelectedDecision.Plugins = nil
	resp.ExecutionTrace.Attempts = make([]looper.AttemptTrace, 100)
	for i := range resp.ExecutionTrace.Attempts {
		resp.ExecutionTrace.Attempts[i] = looper.AttemptTrace{Model: strings.Repeat("x", 256), Usage: usage}
	}
	if wire := router.benchmarkLooperUsage(resp, ctx); len(wire) > maxBenchmarkUsageReceiptBytes || read().Complete {
		t.Fatal("receipt not bounded")
	}
}

func TestBenchmarkCallLimitStopsBeforePromptSelection(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{headers.SRBenchMaxInferenceCalls: "1"}}
	selected := &config.Decision{Name: "paid_selector", Algorithm: &config.AlgorithmConfig{Type: "prompt"}}
	// No selector/client is initialized: reaching model selection would fail.
	_, _, _, model, err := router.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: selected}, "auto", "query", ctx)
	if !errors.Is(err, errBenchmarkCallLimit) || model != "" {
		t.Fatalf("prompt selection was not stopped: model=%q err=%v", model, err)
	}
}

func TestBenchmarkUsageCompressionRequiresLocalScoring(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	for _, method := range []string{"bm25", "embedding", "unknown"} {
		ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: "multi_factor"}, Plugins: []config.DecisionPlugin{{Type: "context_compression", Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "scoring": map[string]interface{}{"method": method}})}}}}
		if got := router.benchmarkUsageScopeKnown(ctx, false); got != (method == "bm25") {
			t.Fatalf("compression scoring %s accounting=%t", method, got)
		}
	}
}
