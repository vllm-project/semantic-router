package extproc

import (
	"strings"
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

func TestStreamingTTFTWaitsForNonEmptyContent(t *testing.T) {
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		start  string
		delta  string
	}{
		{
			name: "responses", format: llmprotocol.OpenAIResponsesV1,
			start: `data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","model":"general","status":"in_progress","output":[]}}` + "\n\n" +
				`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}` + "\n\n" +
				`data: {"type":"response.content_part.added","sequence_number":2,"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"","annotations":[]}}` + "\n\n",
			delta: `data: {"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"hello"}` + "\n\n",
		},
		{
			name: "anthropic", format: llmprotocol.AnthropicMessagesV1,
			start: `data: {"type":"message_start","message":{"id":"msg_1","model":"general","usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n",
			delta: `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"considering"}}` + "\n\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			ctx := &RequestContext{
				Headers: map[string]string{}, SourceFormat: tc.format, TargetFormat: tc.format,
				RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
			}
			router.handleSemanticStreamingResponseBody([]byte(tc.start), false, ctx)
			if ctx.StreamingAborted || ctx.SemanticStreamState == nil || len(ctx.SemanticStreamState.items) != 1 {
				t.Fatal("the stream did not decode its opening item")
			}
			if ctx.TTFTRecorded {
				t.Fatal("the empty start recorded TTFT")
			}
			router.handleSemanticStreamingResponseBody([]byte(tc.delta), false, ctx)
			if ctx.StreamingAborted || !ctx.TTFTRecorded {
				t.Fatal("the first content delta did not record TTFT")
			}
		})
	}
}

// The upstream has answered as soon as the complete mixed Chat frame is
// parsed, even though translation holds the choice to order its two parts.
func TestStreamingTTFTRecordsBeforeMixedChatChoiceIsReleased(t *testing.T) {
	const first = `data: {"id":"r1","model":"general","choices":[{"index":0,"delta":{"role":"assistant","content":"answer marker","reasoning_content":"reasoning marker"}}]}`
	const next = `data: {"id":"r1","model":"general","choices":[{"index":0,"delta":{"content":" continues"}}]}` + "\n\n"
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		for _, path := range []string{"normal", "guarded"} {
			t.Run(string(format)+"/"+path, func(t *testing.T) {
				router := &OpenAIRouter{Config: &config.RouterConfig{}}
				ctx := &RequestContext{
					Headers: map[string]string{}, SourceFormat: format, TargetFormat: llmprotocol.OpenAIChatV1,
					RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
				}
				if path == "guarded" {
					ctx.ResponseGuardHold = &responseGuardHold{guarded: true}
				}
				push := func(body string) string {
					response := router.handleSemanticStreamingResponseBody([]byte(body), false, ctx)
					if ctx.StreamingAborted {
						t.Fatalf("stream aborted after %q", body)
					}
					out := string(clientStreamBytes(response, []byte(body)))
					if hold := ctx.ResponseGuardHold; hold != nil {
						if out != "" {
							t.Fatal("the guarded stream emitted before release")
						}
						return string(hold.frames)
					}
					return out
				}

				push(first)
				if ctx.TTFTRecorded {
					t.Fatal("incomplete frame recorded TTFT")
				}
				opening := push("\n\n")
				if !ctx.TTFTRecorded || ctx.TTFTSeconds <= 0 {
					t.Fatal("complete mixed frame did not record TTFT while held")
				}
				if strings.Contains(opening, "reasoning marker") || strings.Contains(opening, "answer marker") {
					t.Fatalf("held choice emitted before its order was known: %s", opening)
				}
				initial := ctx.TTFTSeconds
				out := push(next)
				reasoning, answer := strings.Index(out, "reasoning marker"), strings.Index(out, "answer marker")
				if reasoning < 0 || answer <= reasoning {
					t.Fatalf("translation must still emit reasoning before the answer: %s", out)
				}
				if ctx.TTFTSeconds != initial {
					t.Fatalf("release changed TTFT from %v to %v", initial, ctx.TTFTSeconds)
				}
			})
		}
	}
}

func TestStreamingTTFTRejectsInvalidMixedChatFrame(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "general", ProcessingStartTime: time.Now().Add(-time.Second),
	}
	body := `data: {"id":"r1","object":"invalid","model":"general","choices":[{"index":0,"delta":{"content":"answer","reasoning_content":"reasoning"}}]}` + "\n\n"
	router.handleSemanticStreamingResponseBody([]byte(body), false, ctx)
	if !ctx.StreamingAborted || ctx.TTFTRecorded {
		t.Fatal("invalid mixed frame must abort without recording TTFT")
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
