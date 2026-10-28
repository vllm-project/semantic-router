package looper

import (
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A stage body does not carry the caller's required capabilities, so a
// text-only helper must be rejected even though its own wire asks for no tools.
func TestLooperCandidateStageAppliesCallerCapabilities(t *testing.T) {
	tools := llmprotocol.Capabilities(llmprotocol.CapabilityTools)
	stageBase := func(caller llmprotocol.CapabilitySet) *Request {
		base := candidateStageRequest()
		worker := base.ModelParams["worker"]
		worker.Capabilities = []string{"text", "tools"}
		base.ModelParams["worker"] = worker
		base.CallerCapabilities = caller
		return base
	}

	client, calls, _ := candidateStageClient(t)
	base := stageBase(tools)
	if err := dispatchCandidateStage(client, base, cloneRequest(base.OriginalRequest), "final"); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("text-only final model admitted for a tools caller: %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("rejected stage reached connector")
	}
	if err := dispatchCandidateStage(client, base, cloneRequest(base.OriginalRequest), "worker"); err != nil || calls.Load() != 1 {
		t.Fatalf("tools-capable worker rejected: err=%v calls=%d", err, calls.Load())
	}

	// Control: without caller capabilities the same final model is admitted.
	base = stageBase(llmprotocol.CapabilitySet{})
	if err := dispatchCandidateStage(client, base, cloneRequest(base.OriginalRequest), "final"); err != nil {
		t.Fatalf("final model rejected without caller capabilities: %v", err)
	}
}

func TestHopClientCarriesCallerCapabilities(t *testing.T) {
	tools := llmprotocol.Capabilities(llmprotocol.CapabilityTools)
	caller := &hopCaller{resp: &graph.HopResponse{Status: 200, Body: []byte(
		`{"object":"chat.completion","model":"panel-a","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`)}}
	var callErr error
	step := stepFunc(func(ctx context.Context, x *graph.Exec, st *graph.State) error {
		_, callErr = NewHopClient(&config.LooperConfig{}, x).CallModelWithOptions(ctx,
			openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}},
			ModelTarget{Name: "panel-a"},
			CallOptions{DecisionName: "remom-route", Iteration: 1, candidateRequest: &Request{CallerCapabilities: tools}},
		)
		st.Response = &graph.Response{}
		return nil
	})
	program := &graph.Program{Name: "test", Steps: graph.Sequence{{ID: "call", Type: "test", Node: step}}}
	if _, err := graph.Run(context.Background(), program, graph.Input{}, graph.Options{Caller: caller}); err != nil {
		t.Fatal(err)
	}
	if callErr != nil {
		t.Fatal(callErr)
	}
	if got := caller.last.Hop.CallerCapabilities; got != tools {
		t.Fatalf("hop caller capabilities = %v, want %v", got.Names(), tools.Names())
	}
}
