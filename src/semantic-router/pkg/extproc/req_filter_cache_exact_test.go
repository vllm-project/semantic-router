package extproc

import (
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const exactCacheHitBody = `{"id":"chatcmpl-cache","object":"chat.completion","created":123,"model":"gpt-4o",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"cached answer"},"finish_reason":"stop"}]}`

func exactCacheHitContext(requestHeaders map[string]string) *RequestContext {
	return &RequestContext{
		RequestID:          "exact-hit",
		Headers:            requestHeaders,
		VSRMatchedKeywords: []string{"prove", "theorem"},
	}
}

func exactCacheHitResponse(t *testing.T, router *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
	t.Helper()

	result := cache.CacheResult{
		ResponseBody: []byte(exactCacheHitBody),
		Found:        true,
		HitKind:      cache.HitKindExact,
		Age:          time.Second,
	}

	response, hit := router.finishExactCacheLookup(ctx, "math", result, nil, time.Millisecond, 1)
	if !hit {
		t.Fatalf("expected the exact lookup to report a hit")
	}
	if ctx.VSRCacheSimilarity != 1 {
		t.Fatalf("exact hit similarity: expected 1, got %v", ctx.VSRCacheSimilarity)
	}
	if response.GetImmediateResponse() == nil {
		t.Fatalf("expected an immediate response for the cache hit")
	}
	return response
}

// TestExactCacheHitEmitsDetailUnderDebug pins the debug-surface half of the
// #2205 contract on the exact fast path (#2911): an exact hit owns similarity
// 1, so a request that opted into x-vsr-debug must see that score inline, along
// with the category and matched keywords the semantic path already emits.
func TestExactCacheHitEmitsDetailUnderDebug(t *testing.T) {
	response := exactCacheHitResponse(t, &OpenAIRouter{},
		exactCacheHitContext(map[string]string{headers.VSRDebug: "true"}))

	if v := immediateHeaderValue(response, headers.VSRCacheHit); v != "true" {
		t.Errorf("cache-hit: expected \"true\", got %q", v)
	}
	if v := immediateHeaderValue(response, "x-vsr-cache-similarity"); v != "1.0000" {
		t.Errorf("cache-similarity: expected \"1.0000\", got %q", v)
	}
	if v := immediateHeaderValue(response, headers.VSRSelectedCategory); v != "math" {
		t.Errorf("selected-category: expected \"math\", got %q", v)
	}
	if v := immediateHeaderValue(response, headers.VSRMatchedKeywords); v != "prove,theorem" {
		t.Errorf("matched-keywords: expected \"prove,theorem\", got %q", v)
	}
}

// TestExactCacheHitDemotesDetailWithoutDebug is the other half of the same
// contract: without the opt-in the intermediate detail stays off the lean
// surface, exactly as it does for a semantic hit.
func TestExactCacheHitDemotesDetailWithoutDebug(t *testing.T) {
	response := exactCacheHitResponse(t, &OpenAIRouter{}, exactCacheHitContext(nil))

	if v := immediateHeaderValue(response, headers.VSRCacheHit); v != "true" {
		t.Errorf("cache-hit: expected \"true\", got %q", v)
	}
	for _, key := range []string{
		"x-vsr-cache-similarity",
		headers.VSRSelectedCategory,
		headers.VSRMatchedKeywords,
	} {
		if v := immediateHeaderValue(response, key); v != "" {
			t.Errorf("lean surface must omit demoted header %q, got %q", key, v)
		}
	}
}
