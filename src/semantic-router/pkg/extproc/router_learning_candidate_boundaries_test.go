package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestLexicographicSurvivorsConstrainActualLearningAndProtection(t *testing.T) {
	for _, stage := range []string{"sampling", "protection"} {
		t.Run(stage, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			params := addTestQuality(router.Config.ModelConfig[primary], .98)
			router.Config.ModelConfig[primary] = params
			router.Config.ModelConfig["outside-band"] = addTestQuality(params, .1)
			router.Config.ModelConfig["survivor"] = addTestQuality(params, .96)
			refs := []config.ModelRef{{Model: "outside-band"}, {Model: primary}, {Model: "survivor"}}
			decision := &config.Decision{
				Name: "synthetic-band", ModelRefs: refs,
				Algorithm: &config.AlgorithmConfig{
					Type: config.DecisionAlgorithmMultiFactor,
					MultiFactor: &config.MultiFactorSelectionConfig{
						Objective: &config.MultiFactorObjectiveConfig{
							Strategy:   config.MultiFactorObjectiveLexicographic,
							Priorities: []config.MultiFactorPriorityConfig{{Factor: config.MultiFactorFactorQuality, Tolerance: .05}},
						},
						Quality: &config.QualityEvidenceConfig{Index: testIntelligenceIndex},
					},
				},
			}
			router.Config.Decisions = []config.Decision{*decision}
			router.Config.RouterLearning = config.RouterLearningConfig{
				Enabled:    true,
				Adaptation: config.RouterLearningAdaptationConfig{Enabled: extprocBoolPtr(stage == "sampling"), CandidateSet: config.RouterLearningCandidateSetGlobal},
				Protection: config.RouterLearningProtectionConfig{Enabled: extprocBoolPtr(stage == "protection"), Scope: config.RouterLearningScopeSession},
			}
			ctx := routerLearningRequestContext("band-session", "band-conversation")
			ctx.VSRSelectedDecision = decision
			if stage == "protection" {
				sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: "band-session", SelectedModel: "outside-band", Timestamp: time.Now()})
			}
			selCtx := &selection.SelectionContext{CandidateModels: refs, DecisionName: decision.Name, SessionID: "band-session"}
			selected, _, err := router.selectModelFromCandidates(selCtx, decision.Algorithm, ctx)
			if err != nil || selected == nil || selected.Model == "outside-band" {
				t.Fatalf("%s escaped objective: selected=%+v err=%v", stage, selected, err)
			}
			assertModelRefs(t, ctx.VSRPolicyEligibleModelRefs, []string{primary, "survivor"})
			assertModelRefs(t, router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetGlobal), []string{primary, "survivor"})
			if stage == "sampling" {
				policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
				if !ok || policy.toReplayAdaptation() == nil || !policy.toReplayAdaptation().Sampling.Used {
					t.Fatal("actual routing sampling was not exercised")
				}
				scores := policy.toReplayAdaptation().Scores
				if len(scores) != 2 {
					t.Fatalf("sampling evaluated outside objective survivors: %+v", scores)
				}
				if _, exists := scores["outside-band"]; exists {
					t.Fatal("excluded quality candidate was scored by adaptation")
				}
			}
		})
	}
}

func TestProtectionRescueRespectsHardRequestOwnership(t *testing.T) {
	for _, boundary := range []string{"portable", "tool_loop", "nonportable"} {
		t.Run(boundary, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
				SessionID:     "rescue-session/rescue-conversation",
				SelectedModel: "cheap", TurnIndex: 1, Timestamp: time.Now(),
			})
			router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
			for index := 0; index < 3; index++ {
				router.routerLearningRuntimeState().recordModelExperience("synthetic-rescue", 0, "cheap", routerLearningOutcomeUnderpowered, 1)
				router.routerLearningRuntimeState().recordModelExperience("synthetic-rescue", 0, "frontier", routerLearningOutcomeGoodFit, 1)
			}
			ctx := routerLearningRequestContext("rescue-session", "rescue-conversation")
			ctx.VSRSelectedDecision = &config.Decision{Name: "synthetic-rescue"}
			switch boundary {
			case "tool_loop":
				ctx.VSRConversationFacts = classification.ConversationFacts{LastAssistantToolCall: true}
			case "nonportable":
				ctx.PreviousResponseID = "synthetic-opaque-response"
			}
			refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			selCtx := &selection.SelectionContext{CandidateModels: refs, DecisionName: "synthetic-rescue"}
			base := &selection.SelectionResult{
				SelectedModel: "frontier", Method: selection.MethodStatic,
				AllScores: map[string]float64{"cheap": .1, "frontier": .9},
			}
			_, result, selected, _, learningErr := router.applyRouterLearning(selCtx, base, &refs[1], ctx)
			if learningErr != nil {
				t.Fatal(learningErr)
			}
			policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
			if !ok {
				t.Fatal("missing protection policy")
			}
			if boundary == "portable" {
				if selected.Model != "frontier" || policy.Action != routerLearningActionRescueSwitch {
					t.Fatalf("portable rescue must remain available: %+v", policy.ToMap())
				}
			} else if selected.Model != "cheap" || result.SessionPolicy == nil || !result.SessionPolicy.HardLocked || policy.Action == routerLearningActionRescueSwitch {
				t.Fatalf("experience overrode %s ownership: model=%s policy=%+v", boundary, selected.Model, policy.ToMap())
			}
		})
	}
}
