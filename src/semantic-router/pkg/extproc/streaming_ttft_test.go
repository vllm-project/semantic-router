package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
