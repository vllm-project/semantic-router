package extproc

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestBenchmarkUsageDirectProofAndCallLimit(t *testing.T) {
	hash := strings.Repeat("a", 64)
	router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: hash}}
	cacheJudge := config.MustStructuredPayload(map[string]interface{}{"enabled": true, "semantic": map[string]interface{}{"match": map[string]interface{}{
		"recall": map[string]interface{}{"top_k": 3}, "judge": map[string]interface{}{"preset": "same_request"},
	}}})
	for _, tc := range []struct {
		name, algorithm, plugin string
		configuration           *config.StructuredPayload
		shadow                  bool
		cache                   bool
		known                   bool
	}{
		{name: "static", algorithm: "static", known: true},
		{name: "multi_factor", algorithm: "multi_factor", known: true},
		{name: "random", algorithm: "random", known: true},
		{name: "prompt helper", algorithm: "prompt"},
		{name: "looper", algorithm: "fusion"},
		{name: "replay", algorithm: "static", plugin: "router_replay", known: true},
		{name: "shadow call", algorithm: "static", plugin: "router_replay", shadow: true},
		{name: "cache recall", algorithm: "static", plugin: "response_cache", known: true},
		{name: "cache judge", algorithm: "static", plugin: "response_cache", configuration: cacheJudge},
		{name: "memory", algorithm: "static", plugin: "memory"},
		{name: "unknown future plugin", algorithm: "static", plugin: "future"},
		{name: "cache hit", algorithm: "static", cache: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := &config.Decision{Algorithm: &config.AlgorithmConfig{Type: tc.algorithm}}
			if tc.plugin != "" {
				d.Plugins = []config.DecisionPlugin{{Type: tc.plugin, Configuration: tc.configuration}}
			}
			ctx := &RequestContext{Headers: map[string]string{headers.VSRBenchExpectedConfigHash: hash, headers.VSRBenchMaxInferenceCalls: "1"}, BenchmarkConfigHash: hash, VSRSelectedDecision: d, VSRSelectedModel: "small", UpstreamStatusCode: 200, VSRCacheHit: tc.cache}
			if tc.shadow {
				ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true, Shadow: &config.RouterReplayShadowConfig{Models: []string{"candidate"}}}
			}
			router.recordBenchmarkInferenceCall(ctx, true)
			if (ctx.BenchmarkInferenceCalls == 1) != tc.known {
				t.Fatalf("unsafe call proof=%d", ctx.BenchmarkInferenceCalls)
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
	ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true, Shadow: &config.RouterReplayShadowConfig{Models: []string{"candidate"}}}
	if read().Complete {
		t.Fatal("untraced shadow calls omitted")
	}
	ctx.RouterReplayPluginConfig = nil
	resp.ExecutionTrace.Attempts = make([]looper.AttemptTrace, 100)
	for i := range resp.ExecutionTrace.Attempts {
		resp.ExecutionTrace.Attempts[i] = looper.AttemptTrace{Model: strings.Repeat("x", 256), Usage: usage}
	}
	if wire := router.benchmarkLooperUsage(resp, ctx); len(wire) > maxBenchmarkUsageReceiptBytes || read().Complete {
		t.Fatal("receipt not bounded")
	}
}

func TestBenchmarkCallLimitStopsBeforeSystemOneSelection(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{headers.VSRBenchMaxInferenceCalls: "1"}}
	selected := &config.Decision{Name: "paid_selector", Algorithm: &config.AlgorithmConfig{Type: "systemone"}}
	// No selector/client is initialized: reaching model selection would fail.
	_, _, _, model, err := router.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: selected}, "vllm-sr/auto", "query", ctx)
	if !errors.Is(err, errBenchmarkCallLimit) || model != "" {
		t.Fatalf("systemone selection was not stopped: model=%q err=%v", model, err)
	}
}

// Only compression that asks no model keeps the receipt's accounting: BM25
// order does, the relevance judge does not.
func TestBenchmarkUsageCompressionRequiresLocalScoring(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	for name, compress := range map[string]map[string]interface{}{
		"bm25":  {},
		"judge": {"match": map[string]interface{}{"judge": map[string]interface{}{"preset": "relevance"}}},
	} {
		ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: "multi_factor"}, Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: compressPayload(compress)}}}}
		if got := router.benchmarkUsageScopeKnown(ctx, false); got != (name == "bm25") {
			t.Fatalf("compression scoring %s accounting=%t", name, got)
		}
	}
}
