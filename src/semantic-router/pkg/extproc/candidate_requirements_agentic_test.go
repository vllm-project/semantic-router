package extproc

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// callerCapabilityRouter is a strict router whose two text models differ only
// in tool support, so a caller requiring tools must change the result.
func callerCapabilityRouter() *OpenAIRouter {
	r := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"text-only":     {Capabilities: []string{"text"}, ContextWindowSize: 20000, MaxOutputTokens: 1024},
		"tools-capable": {Capabilities: []string{"text", "tools"}, ContextWindowSize: 20000, MaxOutputTokens: 1024},
		"reasoner":      {Capabilities: []string{"text", "tools", "reasoning"}, ContextWindowSize: 20000, MaxOutputTokens: 1024},
	}}}}
	r.Config.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
	return r
}

func callerCapabilityRequest() *llmprotocol.Request {
	request := testNeutralRequest("auto", "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(512)
	return request
}

// callerCapabilityContext returns a request context whose accepted envelope
// requires capabilities. No names means no envelope was accepted.
func callerCapabilityContext(d *config.Decision, capabilities ...string) *RequestContext {
	ctx := &RequestContext{SemanticRequest: callerCapabilityRequest(), VSRSelectedDecision: d, TraceContext: context.Background()}
	if len(capabilities) > 0 {
		ctx.AgenticFacts = agenticfacts.Result{Accepted: &agenticfacts.Accepted{RequiredCapabilities: capabilities}}
	}
	return ctx
}

func staticDecision(models ...string) *config.Decision {
	refs := make([]config.ModelRef, 0, len(models))
	for _, model := range models {
		refs = append(refs, config.ModelRef{Model: model})
	}
	return &config.Decision{Name: "agentic", ModelRefs: refs, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic}}
}

func TestStrictSelectionAppliesCallerCapabilities(t *testing.T) {
	r := callerCapabilityRouter()
	d := staticDecision("text-only", "tools-capable")

	// Control: without an envelope the static algorithm takes the first model.
	model, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "hello", "", 1, callerCapabilityContext(d))
	if err != nil || model != "text-only" {
		t.Fatalf("control model=%q err=%v", model, err)
	}

	ctx := callerCapabilityContext(d, "tools")
	model, _, err = r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "hello", "", 1, ctx)
	if err != nil || model != "tools-capable" {
		t.Fatalf("caller tools model=%q err=%v", model, err)
	}
	if len(ctx.VSREligibleModelRefs) != 1 || ctx.VSREligibleModelRefs[0].Model != "tools-capable" {
		t.Fatalf("narrowed pool not carried to later choices: %+v", ctx.VSREligibleModelRefs)
	}
	if refs := r.eligibleLearningModelRefs(d.ModelRefs, ctx); len(refs) != 1 || refs[0].Model != "tools-capable" {
		t.Fatalf("learning pool offered an incapable model: %+v", refs)
	}
}

// Later stages narrow again, so a broken seam could hide behind them. Each
// seam is checked on a fresh context with no earlier narrowing.
func TestCallerCapabilitiesApplyAtEachSeam(t *testing.T) {
	r := callerCapabilityRouter()
	d := staticDecision("text-only", "tools-capable")

	t.Run("decision prefilter", func(t *testing.T) {
		refs, err := r.decisionEligibleModelRefs(d, callerCapabilityContext(d, "tools"))
		if err != nil || len(refs) != 1 || refs[0].Model != "tools-capable" {
			t.Fatalf("refs=%+v err=%v", refs, err)
		}
	})

	t.Run("selection context", func(t *testing.T) {
		input := &selection.SelectionContext{DecisionName: d.Name, CandidateModels: d.ModelRefs}
		filtered, err := r.capabilityEligibleSelectionContext(input, d.Algorithm, callerCapabilityContext(d, "tools"))
		if err != nil || len(filtered.CandidateModels) != 1 || filtered.CandidateModels[0].Model != "tools-capable" {
			t.Fatalf("filtered=%+v err=%v", filtered, err)
		}
	})

	// Learning can draw from a wider pool than the matched decision, so it
	// must apply caller capabilities itself.
	t.Run("learning pool", func(t *testing.T) {
		ctx := callerCapabilityContext(d, "tools")
		refs := r.eligibleLearningModelRefs([]config.ModelRef{{Model: "text-only"}, {Model: "tools-capable"}, {Model: "reasoner"}}, ctx)
		if len(refs) != 2 || refs[0].Model != "tools-capable" || refs[1].Model != "reasoner" {
			t.Fatalf("refs=%+v", refs)
		}
	})
}

// With no capable candidate the request fails closed, and the client response
// names neither a configured model nor the capability the caller asked for.
func TestStrictSelectionFailsClosedOnCallerCapabilities(t *testing.T) {
	r := callerCapabilityRouter()
	d := staticDecision("text-only", "tools-capable")
	ctx := callerCapabilityContext(d, "reasoning")

	_, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "hello", "", 1, ctx)
	if !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("want ErrNoEligibleCandidates, got %v", err)
	}
	var budget *selection.RequestBudgetError
	if errors.As(err, &budget) {
		t.Fatalf("capability failure misclassified as a budget error: %v", err)
	}

	immediate := r.respondSelectionRejected(ctx, "auto", err).GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != 503 {
		t.Fatalf("want HTTP 503, got %+v", immediate)
	}
	body := string(immediate.GetBody())
	for _, secret := range []string{"text-only", "tools-capable", "reasoning"} {
		if strings.Contains(body, secret) {
			t.Fatalf("response body leaks %q: %s", secret, body)
		}
	}
}

// The destination is filtered like any other candidate (CONFIRM-06). When it
// lacks a caller capability, a capable model from the decision's own
// modelRefs serves the request instead.
func TestStrictRouteActionDestinationAppliesCallerCapabilities(t *testing.T) {
	r := callerCapabilityRouter()
	d := staticDecision("tools-capable")
	d.Action = &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "text-only"}

	model, terminal, err := r.decisionRouteActionDestination(d, callerCapabilityContext(d))
	if err != nil || !terminal || model != "text-only" {
		t.Fatalf("control model=%q terminal=%v err=%v", model, terminal, err)
	}

	model, terminal, err = r.decisionRouteActionDestination(d, callerCapabilityContext(d, "tools"))
	if err != nil || !terminal || model != "tools-capable" {
		t.Fatalf("fallback model=%q terminal=%v err=%v", model, terminal, err)
	}

	_, _, err = r.decisionRouteActionDestination(d, callerCapabilityContext(d, "reasoning"))
	if !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("want fail-closed route action, got %v", err)
	}
}

func TestStrictDispatchRechecksCallerCapabilities(t *testing.T) {
	r := callerCapabilityRouter()
	d := staticDecision("text-only", "tools-capable")
	ctx := callerCapabilityContext(d, "tools")
	request := ctx.SemanticRequest

	if err := r.validateDispatchRequirements(request, &providerDispatch{logicalModel: "tools-capable", targetFormat: llmprotocol.OpenAIChatV1}, ctx); err != nil {
		t.Fatalf("capable dispatch rejected: %v", err)
	}
	if err := r.validateDispatchRequirements(request, &providerDispatch{logicalModel: "text-only", targetFormat: llmprotocol.OpenAIChatV1}, ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("incapable dispatch admitted: %v", err)
	}
}

// Caller capabilities join the strict capability check only. Without that
// opt-in they change nothing.
func TestCallerCapabilitiesRequireDeclaredCapabilityMode(t *testing.T) {
	d := staticDecision("text-only", "tools-capable")
	for _, test := range []struct {
		name         string
		requirements *config.CandidateRequirements
	}{
		{"legacy mode", nil},
		{"context limits only", &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}},
	} {
		t.Run(test.name, func(t *testing.T) {
			r := callerCapabilityRouter()
			r.Config.CandidateRequirements = test.requirements
			model, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "hello", "", 1, callerCapabilityContext(d, "tools"))
			if err != nil || model != "text-only" {
				t.Fatalf("model=%q err=%v", model, err)
			}
		})
	}
}

// Caller capabilities can only remove candidates. For any pool and any
// capability list, the eligible set is a subset of the configured pool and of
// the set eligible without an envelope.
func TestCallerCapabilitiesOnlyNarrowCandidates(t *testing.T) {
	r := callerCapabilityRouter()
	request := callerCapabilityRequest()
	models := []string{"text-only", "tools-capable", "reasoner", "undeclared"}
	capabilities := []string{"text", "tools", "reasoning", "image_input", "structured_json"}
	rng := rand.New(rand.NewSource(3379))
	sawNarrowing, sawSurvivor := false, false

	for i := 0; i < 500; i++ {
		var refs []config.ModelRef
		for _, model := range models {
			if rng.Intn(2) == 0 {
				refs = append(refs, config.ModelRef{Model: model})
			}
		}
		var names []string
		for _, capability := range capabilities {
			if rng.Intn(3) == 0 {
				names = append(names, capability)
			}
		}
		caller := (&agenticfacts.Accepted{RequiredCapabilities: names}).Capabilities()

		base, _ := r.eligibleRequestModelRefs(r.Config.CandidateRequirements, refs, request, nil, llmprotocol.CapabilitySet{})
		narrowed, _ := r.eligibleRequestModelRefs(r.Config.CandidateRequirements, refs, request, nil, caller)

		for _, ref := range narrowed {
			if !modelRefInEligibility(ref, refs) {
				t.Fatalf("caller %v introduced %q outside pool %+v", names, ref.Model, refs)
			}
			if !modelRefInEligibility(ref, base) {
				t.Fatalf("caller %v kept %q that the request alone excludes", names, ref.Model)
			}
		}
		sawNarrowing = sawNarrowing || len(narrowed) < len(base)
		sawSurvivor = sawSurvivor || (len(names) > 0 && len(narrowed) > 0)
	}
	// Guards against a vacuous pass where no generated case ever narrowed.
	if !sawNarrowing || !sawSurvivor {
		t.Fatalf("property never exercised: narrowing=%v survivor=%v", sawNarrowing, sawSurvivor)
	}
}
