package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	if ctx.SemanticRequest == nil {
		ctx.SemanticRequest = &llmprotocol.Request{Generation: 1, Model: "model-a"}
	}
	ctx.SemanticRequest.Stream = true
}

// streamAnswer delivers answer as one OpenAI Chat SSE stream through the
// streaming response path, and fails when the stream does not reconstruct a
// terminal answer.
func streamAnswer(t *testing.T, router *OpenAIRouter, ctx *RequestContext, answer string) {
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
		t.Fatal("the stream did not reconstruct a terminal answer")
	}
}

// assertNotEnforced checks that every outcome of the guard says nothing was
// applied to the streamed answer, and that none names an action.
func assertNotEnforced(t *testing.T, outcomes []routerreplay.Outcome) {
	t.Helper()
	for _, outcome := range outcomes {
		if outcome.Target == config.DecisionPluginResponseGuard || outcome.Metadata["direction"] == config.SignalDirectionResponse {
			if outcome.Metadata["enforcement"] != responseGuardStreamingNotEnforced {
				t.Fatalf("outcome = %+v, want enforcement %q", outcome, responseGuardStreamingNotEnforced)
			}
			if action, named := outcome.Metadata["action"]; named {
				t.Fatalf("outcome %+v names action %q, yet nothing was applied to a streamed answer", outcome, action)
			}
		}
	}
}

// A streamed answer is read by the guard and recorded, the way a buffered one
// is. Nothing is applied: its bytes are with the client before the answer
// exists as a whole.
func TestStreamedAnswerIsReadByTheGuardAndNotEnforced(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakScoreServer(t, 0.95, 0.05), "", config.ResponseGuardBlock)
	recorder := startResponseStageReplay(t, router, ctx)

	streamAnswer(t, router, ctx, unsafeAnswer)

	if matched, _ := ctx.ResponseGuard.jailbreakFinding(); !matched {
		t.Fatal("the streamed answer was not scored by the response-direction rule")
	}
	if ctx.ResponseGuard.blocking() != nil || len(ctx.ResponseGuard.warnings()) != 0 {
		t.Fatal("a streamed answer was already delivered, so the guard cannot have acted on it")
	}
	outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
	if signal := outcomeFor(t, outcomes, responseStageSignalKey, ""); signal.Verdict != "detected" {
		t.Fatalf("signal outcome = %+v, want detected", signal)
	}
	assertNotEnforced(t, outcomes)
	sum := sha256.Sum256([]byte(unsafeAnswer))
	for _, outcome := range outcomes {
		if outcome.Source == primaryResponseOutcomeSource && outcome.Metadata["response_sha256"] == hex.EncodeToString(sum[:]) {
			return
		}
	}
	t.Fatalf("outcomes = %+v, want the primary digest of the streamed answer", outcomes)
}

func TestStreamedAnswerIsCheckedForHallucinationAndNotEnforced(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, true, ""))
	recorder := startResponseStageReplay(t, router, ctx)

	streamAnswer(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 1 {
		t.Fatalf("the detector must check a streamed answer once, ran %d time(s)", calls.Load())
	}
	outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
	if signal := outcomeFor(t, outcomes, hallucinationSignalKey, ""); signal.Verdict != "detected" {
		t.Fatalf("signal outcome = %+v, want detected", signal)
	}
	assertNotEnforced(t, outcomes)
}

// A stream that never reaches a terminal answer has nothing to check. The
// detector must not be asked for a partial one.
func TestAbortedStreamIsNotReadByTheGuard(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, false, ""))
	startResponseStageReplay(t, router, ctx)

	expectStreamedResponse(ctx)
	router.handleSemanticStreamingResponseBody([]byte("data: {\"id\":\"response_1\"\n\n"), true, ctx)

	if !ctx.StreamingAborted {
		t.Fatal("the malformed stream was not treated as aborted")
	}
	if calls.Load() != 0 || ctx.ResponseGuard != nil {
		t.Fatalf("an aborted stream has no terminal answer, yet the guard read it (%d detector calls)", calls.Load())
	}
}
