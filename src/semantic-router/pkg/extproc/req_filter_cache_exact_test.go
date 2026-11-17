package extproc

import (
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
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

// An exact hit owns similarity 1. A request allowed the debug headers sees
// it in the trace with the matched keywords; every caller sees the cache
// path and the decision whose cache answered.
func TestExactCacheHitReportsItsSimilarityInTheTrace(t *testing.T) {
	ctx := exactCacheHitContext(map[string]string{headers.VSRDebug: "true"})
	response := exactCacheHitResponse(t, &OpenAIRouter{}, ctx)
	appendRoutedResponseHeaders(response, ctx)

	set := response.GetImmediateResponse().GetHeaders().GetSetHeaders()
	if v := headerValuesByName(set)[headers.VSRResponsePath]; v != headers.ResponsePathCache {
		t.Errorf("response path: expected cache, got %q", v)
	}
	if v := headerValuesByName(set)[headers.VSRSelectedDecision]; v != "math" {
		t.Errorf("selected decision: expected math, got %q", v)
	}
	trace := traceOf(t, set)
	if trace[headers.TraceCacheSimilarity] != "1.0" || trace[headers.TraceSignals] != `("keyword:prove" "keyword:theorem")` {
		t.Errorf("trace = %v, want similarity 1.0 and the matched keywords", trace)
	}
}

// Without the opt-in the trace stays off the response, exactly as it does
// for a semantic hit.
func TestExactCacheHitKeepsTheTraceFromOtherCallers(t *testing.T) {
	ctx := exactCacheHitContext(nil)
	response := exactCacheHitResponse(t, &OpenAIRouter{}, ctx)
	appendRoutedResponseHeaders(response, ctx)

	if trace := traceOf(t, response.GetImmediateResponse().GetHeaders().GetSetHeaders()); trace != nil {
		t.Errorf("a caller without debug got the trace %v", trace)
	}
}

// Replay keeps what the client received from an exact hit, once, the same way
// a semantic hit records it.
func TestExactCacheHitRecordsTheAnswerTheClientReceived(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
		recorder.SetCapturePolicy(true, true, 1<<20)
		ctx := exactCacheHitContext(nil)
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
		ctx.ExpectStreamingResponse = streaming
		ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true, PromptCapture: config.PromptCaptureRaw, CaptureRequestBody: true, CaptureResponseBody: true}
		ctx.SemanticRequest = testNeutralRequest("gpt-4o", "Prove the theorem.")
		ctx.SemanticRequest.Stream = streaming

		client := exactCacheHitResponse(t, &OpenAIRouter{ReplayRecorder: recorder}, ctx).GetImmediateResponse().GetBody()
		record, found := recorder.GetRecord(ctx.RouterReplayID)
		require.True(t, found)
		require.Equal(t, string(client), record.ResponseBody, "streaming=%t", streaming)
		types := make([]string, 0, len(record.ToolTrace.Steps))
		for _, step := range record.ToolTrace.Steps {
			types = append(types, step.Type)
		}
		require.Equal(t, []string{replayToolStepUserInput, replayToolStepAssistantFinalResponse}, types, "streaming=%t", streaming)
	}
}
