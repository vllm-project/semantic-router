package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// TTFT is the model's first response, not the gateway's first frame: a
// keepalive or comment frame that arrives while the model queues must not
// become the recorded TTFT.
func TestStreamingTTFTIgnoresKeepaliveFrames(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
	}
	router.initializeSemanticResponseStream(ctx)

	keepalive := "data: {\"type\":\"ping\"}\n\n"
	router.handleSemanticStreamingResponseBody([]byte(keepalive), false, ctx)
	if ctx.TTFTRecorded {
		t.Fatalf("a keepalive frame recorded TTFT %.4fs", ctx.TTFTSeconds)
	}

	first := "data: {\"id\":\"r1\",\"model\":\"general\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n"
	router.handleSemanticStreamingResponseBody([]byte(first), false, ctx)
	if !ctx.TTFTRecorded {
		t.Fatal("the first content chunk recorded no TTFT")
	}
	if ctx.TTFTSeconds <= 0 || ctx.TTFTSeconds > 5 {
		t.Fatalf("TTFT = %.4fs, want the wait since processing started", ctx.TTFTSeconds)
	}
}

// A body that carries both the answer and its end settles the stream in one
// read: the turn it records must already know the latency.
func TestStreamingTTFTSurvivesSingleReadSettle(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		SessionID: "single-read", RequestID: "single-read",
		RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
	}
	gate := selection.DefaultProgressGateConfig()
	gate.Enabled = true
	configureProgressEvidence(ctx, gate, time.Now())

	body := "data: {\"id\":\"r1\",\"model\":\"general\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n" +
		"data: {\"id\":\"r1\",\"model\":\"general\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n" +
		"data: [DONE]\n\n"
	router.handleSemanticStreamingResponseBody([]byte(body), true, ctx)

	window := sessiontelemetry.RecentTurnOutcomes(routingLearningStateKey(ctx), time.Now())
	if len(window) != 1 {
		t.Fatalf("the settled stream recorded %d turns, want 1", len(window))
	}
	if !window[0].LatencyKnown || window[0].LatencyMs <= 0 {
		t.Fatalf("the recorded turn lost its latency: %+v", window[0])
	}
}

// A held stream is the model answering while the guard reads it, so it records
// TTFT too, whichever way the release goes.
func TestHeldStreamingTTFTRecordsWhileHeld(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
		ResponseGuardHold: &responseGuardHold{},
	}
	first := "data: {\"id\":\"r1\",\"model\":\"general\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n"
	router.handleHeldStreamBody([]byte(first), false, ctx)
	if !ctx.TTFTRecorded {
		t.Fatal("a held stream recorded no TTFT")
	}
}

// A stream the provider aborts before any content still records no TTFT.
func TestStreamingTTFTStaysUnrecordedWithoutContent(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
	}
	router.initializeSemanticResponseStream(ctx)
	router.handleSemanticStreamingResponseBody([]byte(": ping\n\n"), true, ctx)
	if ctx.TTFTRecorded {
		t.Fatal("a stream with no content recorded a TTFT")
	}
}
