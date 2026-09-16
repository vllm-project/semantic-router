package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func strictCandidateRouter() *OpenAIRouter {
	r := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"text":    {Capabilities: []string{"chat"}, ContextWindowSize: 20000, MaxOutputTokens: 1024},
		"vision":  {Capabilities: []string{"chat", "vision", "tools"}, ContextWindowSize: 20000, MaxOutputTokens: 1024},
		"small":   {Capabilities: []string{"chat", "vision", "tools"}, ContextWindowSize: 8192, MaxOutputTokens: 1024},
		"unknown": {},
	}}}}
	r.Config.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
	return r
}

func strictCandidateRequest() *llmprotocol.Request {
	return &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "describe"}, {Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"}}}}, Sampling: llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(512)}}
}

func TestStrictCandidatesAgreeInRuntimePreviewAndAdaptation(t *testing.T) {
	r := strictCandidateRouter()
	request := strictCandidateRequest()
	d := &config.Decision{Name: "reasoning", ModelRefs: []config.ModelRef{{Model: "text"}, {Model: "small"}, {Model: "unknown"}, {Model: "vision"}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic}}
	ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: d, TraceContext: context.Background()}
	model, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "describe", "", 1, ctx)
	if err != nil || model != "vision" {
		t.Fatalf("runtime model=%q err=%v", model, err)
	}
	preview := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, Demand: selection.DemandForRequest(request)})
	if preview.SelectedModel != model || preview.Status != services.EvalSelectionSelected {
		t.Fatalf("preview=%+v", preview)
	}
	if len(ctx.VSREligibleModelRefs) != 1 || len(ctx.VSRPolicyEligibleModelRefs) != 1 {
		t.Fatal("prefilter was not carried to later choices")
	}
	refs := r.eligibleLearningModelRefs(d.ModelRefs, ctx)
	if len(refs) != 1 || refs[0].Model != "vision" {
		t.Fatalf("adaptation resurrected candidate: %+v", refs)
	}
	d.Algorithm.MinimumCandidates = 2
	if _, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "describe", "", 1, ctx); err == nil {
		t.Fatal("runtime ignored minimum pool")
	}
	if got := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, Demand: selection.DemandForRequest(request)}); got.Status != services.EvalSelectionUnavailable {
		t.Fatalf("preview minimum=%+v", got)
	}
}

func TestStrictDispatchRechecksActualGrowthAndPermissionScope(t *testing.T) {
	r := strictCandidateRouter()
	request := strictCandidateRequest()
	d := &config.Decision{Name: "reasoning", ModelRefs: []config.ModelRef{{Model: "vision"}, {Model: "small"}}}
	ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: d, VSREligibleModelRefs: []config.ModelRef{{Model: "vision"}}}
	dispatch := &providerDispatch{logicalModel: "vision", targetFormat: llmprotocol.OpenAIChatV1}
	if err := r.validateDispatchRequirements(request, dispatch, ctx); err != nil {
		t.Fatal(err)
	}
	if err := r.validateDispatchRequirements(request, &providerDispatch{logicalModel: "text"}, ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("scope err=%v", err)
	}
	request.Messages[0].Content = append(request.Messages[0].Content, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: string(make([]byte, 50000))})
	if err := r.validateDispatchRequirements(request, dispatch, ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("growth err=%v", err)
	}
	if got := r.findQualifiedRerouteModel(d, dispatch, llmprotocol.RequiredCapabilities(*request), ctx); got != "" {
		t.Fatalf("fallback resurrected %q", got)
	}
}

func TestStrictOmittedOutputPolicyMatchesEncodedDispatch(t *testing.T) {
	r := strictCandidateRouter()
	payload, _ := config.NewStructuredPayload(map[string]any{"default_max_tokens": 64, "max_tokens_limit": 100})
	d := &config.Decision{Name: "simple", ModelRefs: []config.ModelRef{{Model: "text"}}, Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}}}
	for _, test := range []struct {
		name   string
		output *int64
		want   int64
	}{{"omitted", nil, 64}, {"explicit", llmprotocol.Int64(80), 80}, {"capped", llmprotocol.Int64(200), 100}} {
		t.Run(test.name, func(t *testing.T) {
			request := &llmprotocol.Request{Model: "text", Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}}}, Sampling: llmprotocol.Sampling{MaxOutputTokens: test.output}}
			demand, err := selection.EffectiveCandidateDemand(request, d)
			if err != nil {
				t.Fatal(err)
			}
			if demand.MaxOutputTokens == nil || *demand.MaxOutputTokens != test.want {
				t.Fatalf("prefilter demand=%+v", demand)
			}
			if _, err = r.applySemanticRequestParams(d, request, config.DefaultRecipeName); err != nil {
				t.Fatal(err)
			}
			body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(*request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatal(err)
			}
			actual, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatal(err)
			}
			if actual.Sampling.MaxOutputTokens == nil || *actual.Sampling.MaxOutputTokens != test.want {
				t.Fatalf("outbound output=%v", actual.Sampling.MaxOutputTokens)
			}
			if selection.DemandForRequest(&actual).InputTokens != demand.InputTokens {
				t.Fatal("prefilter and outbound count disagree")
			}
		})
	}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}}}}
	noPolicy := &config.Decision{Name: "simple", ModelRefs: []config.ModelRef{{Model: "text"}}}
	if got := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: noPolicy, Demand: selection.DemandForRequest(request)}); got.Status != services.EvalSelectionUnavailable {
		t.Fatalf("invented provider default: %+v", got)
	}
}

func TestStrictProviderGrowthReroutesOnlyWithinRetainedPolicy(t *testing.T) {
	for _, allowLarge := range []bool{false, true} {
		r, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
		r.Config.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
		params := r.Config.ModelConfig[primary]
		params.Capabilities = []string{"chat"}
		params.ContextWindowSize = 100
		params.MaxOutputTokens = 64
		r.Config.ModelConfig[primary] = params
		params.ContextWindowSize = 200
		r.Config.ModelConfig["large"] = params
		d := &config.Decision{Name: "reasoning", ModelRefs: []config.ModelRef{{Model: primary}, {Model: "large"}}}
		request := testNeutralRequest(primary, "hello")
		request.Sampling.MaxOutputTokens = llmprotocol.Int64(64)
		ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
		ctx.VSRSelectedDecision = d
		if _, err := r.decisionEligibleModelRefs(d, ctx); err != nil {
			t.Fatal(err)
		}
		if !allowLarge {
			ctx.VSREligibleModelRefs = []config.ModelRef{{Model: primary}}
			ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(ctx.VSREligibleModelRefs)
		}
		request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: string(make([]byte, 180))}}})
		dispatch, err := r.prepareProviderDispatch(request, primary, d.Name, false, ctx)
		if allowLarge {
			if err != nil || dispatch == nil || dispatch.logicalModel != "large" {
				t.Fatalf("permitted recovery dispatch=%+v err=%v", dispatch, err)
			}
		} else if !errors.Is(err, selection.ErrNoEligibleCandidates) || dispatch != nil {
			t.Fatalf("excluded fallback resurrected dispatch=%+v err=%v", dispatch, err)
		}
	}
}

func TestStrictAlgorithmHelperCannotRerouteIntoWorker(t *testing.T) {
	r, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	r.Config.CandidateRequirements = &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}
	params := r.Config.ModelConfig[primary]
	params.ContextWindowSize = 40
	params.MaxOutputTokens = 32
	r.Config.ModelConfig[primary] = params
	params.ContextWindowSize = 1000
	r.Config.ModelConfig["worker"] = params
	d := &config.Decision{Name: "agent", ModelRefs: []config.ModelRef{{Model: "worker"}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: primary}}}
	request := testNeutralRequest(primary, string(make([]byte, 100)))
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(32)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = d
	ctx.LooperRequest = true
	if dispatch, err := r.prepareProviderDispatch(request, primary, d.Name, false, ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) || dispatch != nil {
		t.Fatalf("helper was rewritten: %+v %v", dispatch, err)
	}
}

func TestSelectorHardFiltersMustRetainMinimumPoolInPreviewAndRuntime(t *testing.T) {
	r := strictCandidateRouter()
	for model, score := range map[string]float64{"text": .95, "vision": .2} {
		p := r.Config.ModelConfig[model]
		p = addTestQuality(p, score)
		r.Config.ModelConfig[model] = p
	}
	floor := 70.0
	d := &config.Decision{Name: "reasoning", ModelRefs: []config.ModelRef{{Model: "text"}, {Model: "vision"}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MinimumCandidates: 2, MultiFactor: &config.MultiFactorSelectionConfig{Weights: &config.MultiFactorWeightsConfig{Quality: 1}, Quality: &config.QualityEvidenceConfig{Index: testIntelligenceIndex, MinScore: &floor}, OnNoCandidates: "fail"}}}
	request := testNeutralRequest("text", "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(64)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = d
	if _, _, err := r.selectDecisionRuntimeModel(&decision.DecisionResult{Decision: d}, d.Name, "hello", "", 1, ctx); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("runtime minimum after hard filter: %v", err)
	}
	result := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, Demand: selection.DemandForRequest(request)})
	if result.Status != services.EvalSelectionUnavailable || result.SelectedModel != "" {
		t.Fatalf("Preview admitted insufficient hard-filtered pool: %+v", result)
	}
}

func TestStrictRouteActionUsesSameCandidateDemand(t *testing.T) {
	r := strictCandidateRouter()
	request := strictCandidateRequest()
	d := &config.Decision{Name: "guarded", Action: &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "text"}, ModelRefs: []config.ModelRef{{Model: "vision"}}}
	ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: d}
	model, terminal, err := r.decisionRouteActionDestination(d, ctx)
	if err != nil || !terminal || model != "vision" {
		t.Fatalf("action=%q %v %v", model, terminal, err)
	}
	result := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, Demand: selection.DemandForRequest(request)})
	if result.SelectedModel != model {
		t.Fatalf("Preview action mismatch: %+v", result)
	}
}
