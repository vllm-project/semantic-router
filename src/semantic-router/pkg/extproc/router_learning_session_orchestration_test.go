package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Exercise the production preflight -> adaptation -> switch orchestration with
// maintained messages and actual selections, including a bypass control.
func TestRouterLearningSessionOrchestrationSuppressesActualSampling(t *testing.T) {
	corpus, _ := loadProtectionCorpus(t)
	for _, scenario := range corpus.Scenarios {
		if scenario.ID != "tool-loop-and-release" {
			continue
		}
		for _, mode := range []string{"apply", "bypass"} {
			t.Run(mode, func(t *testing.T) {
				scenario.Mode = mode
				runProtectionOrchestration(t, scenario)
			})
		}
		return
	}
	t.Fatal("missing tool-loop orchestration fixture")
}

func runProtectionOrchestration(t *testing.T, scenario protectionScenario) {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	calls := 0
	original := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { calls++; return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = original })
	cfg := routerLearningTestConfig(scenario.Scope)
	cfg.DefaultModel = "protection-cheap"
	cfg.ModelConfig = map[string]config.ModelParams{"protection-cheap": {}, "protection-frontier": {}}
	router := &OpenAIRouter{Config: cfg}
	request := &llmprotocol.Request{}
	for turn, step := range scenario.Steps {
		for _, message := range step.Messages {
			request.Messages = append(request.Messages, protectionNeutralMessage(message))
		}
		input := protectionScenarioInput(router, scenario, step, turn, request)
		before := calls
		ctx, result, ref, _ := router.applyRouterLearning(input.selCtx, input.baseResult, input.selectedModelRef, input.ctx)
		wantSampling := scenario.Mode == "bypass" || step.Expected.Sampling
		assertAdaptationSampled(t, input.ctx, wantSampling)
		if (calls > before) != wantSampling {
			t.Fatalf("%s: sampling invocation does not match permission", step.ID)
		}
		if scenario.Mode == "apply" && step.Expected.Category == "blocked" {
			previous := input.selCtx.AgenticSession.PreviousModel
			if previous == "" || ref.Model != previous || result.SelectedModel != previous {
				t.Fatalf("%s: protected continuation switched from %q to %q", step.ID, previous, ref.Model)
			}
		}
		recordAgenticSessionDecision(ctx, result, ref, input.ctx)
	}
}
