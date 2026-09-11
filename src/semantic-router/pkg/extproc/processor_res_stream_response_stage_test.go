package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// expectStreamedResponse puts the request in the state Envoy leaves it in when
// the client asked for a stream and the backend answers with one.
func expectStreamedResponse(ctx *RequestContext) {
	ctx.IsStreamingResponse = true
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	ctx.RequestModel = "model-a"
	ctx.SemanticRequest = &llmprotocol.Request{Generation: 1, Model: "model-a", Stream: true}
}

// streamResponseStageAnswer delivers answer as one OpenAI Chat SSE stream
// through the streaming response path, and fails when the stream does not
// reconstruct a terminal answer to observe.
func streamResponseStageAnswer(t *testing.T, router *OpenAIRouter, ctx *RequestContext, answer string) {
	t.Helper()
	expectStreamedResponse(ctx)

	delta, err := json.Marshal(answer)
	if err != nil {
		t.Fatalf("encode stream delta: %v", err)
	}
	body := strings.Join([]string{
		`data: {"id":"response_1","model":"model-a","choices":[{"index":0,"delta":{"role":"assistant","content":` +
			string(delta) + `},"finish_reason":"stop"}]}` + "\n\n",
		`data: {"id":"response_1","model":"model-a","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n",
		"data: [DONE]\n\n",
	}, "")

	router.handleSemanticStreamingResponseBody([]byte(body), true, ctx)
	if ctx.SemanticResponse == nil {
		t.Fatal("the stream did not reconstruct a terminal answer to observe")
	}
}

// assertStreamedOutcome checks the one outcome a streamed response leaves: the
// verdict it observed, and that nothing enforced it.
func assertStreamedOutcome(t *testing.T, outcomes []routerreplay.Outcome, target, verdict string) {
	t.Helper()
	if len(outcomes) != 1 {
		t.Fatalf("outcomes = %+v, want one per declared rule", outcomes)
	}
	outcome := outcomes[0]
	if outcome.Target != target || outcome.Verdict != verdict {
		t.Fatalf("outcome = %+v, want verdict %q under %s", outcome, verdict, target)
	}
	if outcome.Metadata["enforcement"] != responseStageStreamingNotEnforced {
		t.Fatalf("enforcement = %q, want %q", outcome.Metadata["enforcement"], responseStageStreamingNotEnforced)
	}
	if action, named := outcome.Metadata["action"]; named {
		t.Fatalf("the record named action %q, yet no plugin ran on a streamed response", action)
	}
}

// A streamed response is scored by the declared response-direction rule and
// recorded, the way a buffered one is. It is not enforced: the bytes are with
// the client before the terminal answer exists, so the decision's
// response_jailbreak plugin never runs and the record names no action.
func TestStreamedResponseJailbreakIsObservedAndNotEnforced(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "block")
	recorder := startResponseStageReplay(t, router, ctx)

	streamResponseStageAnswer(t, router, ctx, content)

	if len(ctx.VSRMatchedResponseJailbreak) != 1 || ctx.VSRMatchedResponseJailbreak[0] != responseStageRuleName {
		t.Fatalf("matched response rules = %v, want [%s] (errors=%v)",
			ctx.VSRMatchedResponseJailbreak, responseStageRuleName, ctx.VSRSignalErrors)
	}
	if score := ctx.VSRSignalConfidences[responseStageSignalKey]; score < 0.9 {
		t.Fatalf("the observation must carry the score it thresholded, got %v", score)
	}
	if ctx.ResponseJailbreakDetected {
		t.Fatal("a streamed response was already delivered, so the block action must not have run")
	}
	assertStreamedOutcome(t, replayOutcomes(t, recorder, ctx.RouterReplayID), responseStageSignalKey, "detected")
}

// The same for the hallucination rule: the streamed answer is checked against
// the grounding context the request carried and the verdict is recorded, while
// the decision's hallucination plugin takes no action on it.
func TestStreamedAnswerIsCheckedForHallucinationAndNotEnforced(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")
	recorder := startResponseStageReplay(t, router, ctx)

	streamResponseStageAnswer(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 1 {
		t.Fatalf("the detector must check a streamed answer once, ran %d time(s)", calls.Load())
	}
	if len(ctx.VSRMatchedHallucination) != 1 || ctx.VSRMatchedHallucination[0] != hallucinationRuleName {
		t.Fatalf("matched hallucination rules = %v, want [%s] (errors=%v)",
			ctx.VSRMatchedHallucination, hallucinationRuleName, ctx.VSRSignalErrors)
	}
	if ctx.HallucinationDetected || ctx.UnverifiedFactualResponse {
		t.Fatalf("a streamed answer is observed, not enforced: detected=%v unverified=%v",
			ctx.HallucinationDetected, ctx.UnverifiedFactualResponse)
	}
	assertStreamedOutcome(t, replayOutcomes(t, recorder, ctx.RouterReplayID), hallucinationSignalKey, "detected")
}

// A stream that never reaches a terminal answer has nothing to check. The
// detector must not be asked for a partial one, and no verdict is recorded for
// text the client may never have received in full.
func TestAbortedStreamIsNotCheckedForHallucination(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")
	recorder := startResponseStageReplay(t, router, ctx)

	expectStreamedResponse(ctx)
	router.handleSemanticStreamingResponseBody([]byte("data: {\"id\":\"response_1\"\n\n"), true, ctx)

	if !ctx.StreamingAborted {
		t.Fatal("the malformed stream was not treated as aborted")
	}
	if calls.Load() != 0 {
		t.Fatalf("an aborted stream has no terminal answer, yet the detector was asked %d time(s)", calls.Load())
	}
	if outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID); len(outcomes) != 0 {
		t.Fatalf("outcomes = %+v, want none for a stream that never answered", outcomes)
	}
}
