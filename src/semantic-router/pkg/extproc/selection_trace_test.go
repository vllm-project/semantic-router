package extproc

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestSelectionTraceDoesNotSurviveAnotherSelectionAttempt(t *testing.T) {
	for _, path := range []string{"single", "fallback", "rejected", "cancelled"} {
		t.Run(path, func(t *testing.T) {
			candidates := []config.ModelRef{{Model: "a"}, {Model: "b"}}
			router := &OpenAIRouter{}
			ctx := &RequestContext{VSRSelectionTrace: &selectiontrace.MultiFactorObjective{
				FinalSurvivors: []config.ModelRef{{Model: "prior"}},
			}}
			switch path {
			case "single":
				candidates = candidates[:1]
			case "rejected":
				registry := selection.NewRegistry()
				registry.Register(selection.MethodStatic, selectionResultSelector{err: selection.ErrNoEligibleCandidates})
				router.ModelSelector = registry
			case "cancelled":
				request, cancel := context.WithCancel(context.Background())
				cancel()
				ctx.TraceContext = request
			}
			selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: candidates}, nil, ctx)
			if (path == "rejected" || path == "cancelled") && (err == nil || selected != nil) {
				t.Fatalf("expected failed selection: %+v, %v", selected, err)
			}
			if ctx.VSRSelectionTrace != nil {
				t.Fatal("selection reused the previous attempt's objective trace")
			}
		})
	}
}

func TestSelectionTracePreviewAndLiveLatencyCoverage(t *testing.T) {
	for _, learning := range []bool{false, true} {
		for _, coverage := range []string{"none", "partial", "complete"} {
			name := coverage
			if learning {
				name += "_learning"
			}
			t.Run(name, func(t *testing.T) {
				expensive, cheap := t.Name()+"/expensive", t.Name()+"/cheap"
				t.Cleanup(func() {
					latency.RemoveModelFromTTFTCache(expensive)
					latency.RemoveModelFromTTFTCache(cheap)
				})
				a, b := modelParamsWithTestQuality(.9), modelParamsWithTestQuality(.9)
				a.Pricing = config.ModelPricing{Currency: "USD", PromptPer1M: 4, CompletionPer1M: 4}
				b.Pricing = config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 1}
				router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
					ModelConfig: map[string]config.ModelParams{expensive: a, cheap: b},
				}}}
				if learning {
					router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation).RouterLearning
				}
				switch coverage {
				case "partial":
					latency.UpdateTTFT(expensive, 200)
				case "complete":
					latency.UpdateTTFT(expensive, 1)
					latency.UpdateTTFT(cheap, 200)
				}
				decision := &config.Decision{
					Name: "coverage", ModelRefs: []config.ModelRef{{Model: expensive}, {Model: cheap}},
					Algorithm: &config.AlgorithmConfig{
						Type: config.DecisionAlgorithmMultiFactor,
						MultiFactor: &config.MultiFactorSelectionConfig{LatencyMetric: "ttft", Objective: &config.MultiFactorObjectiveConfig{
							Strategy:   config.MultiFactorObjectiveLexicographic,
							Priorities: []config.MultiFactorPriorityConfig{{Factor: "quality"}, {Factor: "latency"}, {Factor: "cost"}},
						}},
					},
				}
				preview := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision, ContextTokenCount: 128})
				ctx := &RequestContext{VSRSelectedDecision: decision}
				selCtx := router.buildSelectionContext(decision.ModelRefs, decision.Name, "", decision.Algorithm, "", nil, ctx)
				selCtx.InputTokens = 128
				selected, _, err := router.selectModelFromCandidates(selCtx, decision.Algorithm, ctx)
				want := cheap
				if coverage == "complete" {
					want = expensive
				}
				if err != nil || selected == nil || selected.Model != want || preview.SelectedModel != want {
					t.Fatalf("live=%+v err=%v preview=%+v want=%s", selected, err, preview, want)
				}
				if preview.MultiFactor == nil || !reflect.DeepEqual(preview.MultiFactor, ctx.VSRSelectionTrace) {
					t.Fatalf("preview/live objective evidence differs: %+v / %+v", preview.MultiFactor, ctx.VSRSelectionTrace)
				}
				stage := preview.MultiFactor.Stages[1]
				wantAction, wantAvailable := selectiontrace.StageSkipped, 0
				switch coverage {
				case "partial":
					wantAvailable = 1
				case "complete":
					wantAction, wantAvailable = selectiontrace.StageApplied, 2
				}
				if stage.Action != wantAction || stage.Available != wantAvailable || stage.Total != 2 {
					t.Fatalf("latency evidence=%+v", stage)
				}
				recorded := buildReplayRouteDiagnostics(ctx, "auto", selected.Model, decision.Name, 0, 0)
				if !reflect.DeepEqual(recorded.SelectionTrace, preview.MultiFactor) {
					t.Fatal("replay lost the evaluated objective stages")
				}
				recorded.SelectionTrace.Stages[0].Available = 0
				if ctx.VSRSelectionTrace.Stages[0].Available != 2 {
					t.Fatal("replay trace aliases mutable request evidence")
				}
			})
		}
	}
}
