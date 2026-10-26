package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// remomCallerDecision has a tools-capable worker and an explicitly configured
// text-only synthesis model, so only the caller's capabilities exclude it.
func remomCallerDecision() *config.Decision {
	return &config.Decision{
		Name:      "agentic-remom",
		ModelRefs: []config.ModelRef{{Model: "tools-capable"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{
			BreadthSchedule: []int{1}, ModelDistribution: "round_robin", SynthesisModel: "text-only",
		}},
	}
}

// A hop never reads the envelope. It serves the set its client request
// accepted, which the Router carries on the typed hop.
func TestStrictHopDispatchAppliesCarriedCallerCapabilities(t *testing.T) {
	r := callerCapabilityRouter()
	d := remomCallerDecision()
	hopContext := func(caller llmprotocol.CapabilitySet) *RequestContext {
		ctx := callerCapabilityContext(d)
		ctx.Hop = &routing.Hop{Decision: d.Name, CallerCapabilities: caller}
		ctx.LooperRequest = true
		return ctx
	}
	dispatch := func(model string) *providerDispatch {
		return &providerDispatch{logicalModel: model, targetFormat: llmprotocol.OpenAIChatV1}
	}
	tools := llmprotocol.Capabilities(llmprotocol.CapabilityTools)

	ctx := hopContext(tools)
	if err := r.validateDispatchRequirements(ctx.SemanticRequest, dispatch("text-only"), ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("text-only synthesis hop admitted for a tools caller: %v", err)
	}
	if err := r.validateDispatchRequirements(ctx.SemanticRequest, dispatch("tools-capable"), ctx); err != nil {
		t.Fatalf("tools-capable worker hop rejected: %v", err)
	}

	// Control: the helper is in scope, so without caller capabilities it passes.
	ctx = hopContext(llmprotocol.CapabilitySet{})
	if err := r.validateDispatchRequirements(ctx.SemanticRequest, dispatch("text-only"), ctx); err != nil {
		t.Fatalf("permitted helper rejected without caller capabilities: %v", err)
	}
}

// Regression for the review: a strict ReMoM decision must not dispatch its
// text-only synthesis model for a caller that requires tools. With on_error
// "fail" the request fails closed; with the default "skip" ReMoM keeps its
// existing fallback to the previous round, which a capable worker served.
func TestStrictLooperRejectsIncapableSynthesisModel(t *testing.T) {
	var mu sync.Mutex
	var called []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(req.Body).Decode(&body)
		mu.Lock()
		called = append(called, body.Model)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"stage","object":"chat.completion","created":1,"model":"` + body.Model + `","choices":[{"index":0,"message":{"role":"assistant","content":"Answer."},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(server.Close)

	run := func(onError string, capabilities ...string) (*looper.Response, []string, error) {
		mu.Lock()
		called = nil
		mu.Unlock()
		r := callerCapabilityRouter()
		d := remomCallerDecision()
		d.Algorithm.ReMoM.OnError = onError
		ctx := callerCapabilityContext(d, capabilities...)
		if _, err := r.decisionEligibleModelRefs(d, ctx); err != nil {
			t.Fatalf("worker selection: %v", err)
		}
		req, response := r.buildLooperRequest(ctx.SemanticRequest, d, ctx)
		if response != nil || req == nil {
			t.Fatalf("build looper request: %+v", response)
		}
		resp, err := looper.NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
		mu.Lock()
		defer mu.Unlock()
		return resp, append([]string(nil), called...), err
	}
	dispatched := func(models []string, model string) bool {
		for _, called := range models {
			if called == model {
				return true
			}
		}
		return false
	}

	t.Run("on_error fail rejects the request", func(t *testing.T) {
		_, models, err := run(config.ReMoMOnErrorFail, "tools")
		if !errors.Is(err, selection.ErrNoEligibleCandidates) {
			t.Fatalf("want rejection of the text-only synthesis model, got %v", err)
		}
		if dispatched(models, "text-only") {
			t.Fatalf("text-only synthesis model was dispatched: %v", models)
		}
	})

	t.Run("default skip answers from the capable worker", func(t *testing.T) {
		resp, models, err := run("", "tools")
		if err != nil {
			t.Fatalf("skip run failed: %v", err)
		}
		if dispatched(models, "text-only") {
			t.Fatalf("text-only synthesis model was dispatched: %v", models)
		}
		if resp.Model != "tools-capable" {
			t.Fatalf("final answer came from %q, want tools-capable", resp.Model)
		}
	})

	// Control: without an envelope the same decision completes through its
	// synthesis model, so the rejections above come from the caller only.
	t.Run("control without envelope", func(t *testing.T) {
		resp, models, err := run(config.ReMoMOnErrorFail)
		if err != nil {
			t.Fatalf("control run failed: %v", err)
		}
		if !dispatched(models, "text-only") || resp.Model != "text-only" {
			t.Fatalf("control did not finish on the synthesis model: model=%q calls=%v", resp.Model, models)
		}
	})
}
