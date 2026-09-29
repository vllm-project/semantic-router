package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestMaterializedResponseHistoryIsPortableBeforeSelection(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	storage, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := storage.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	stored := &responseapi.StoredResponse{
		ID: "resp_synthetic_parent", Object: "response", Model: "cheap", Status: "completed",
		Input:  []responseapi.InputItem{{Type: "message", Role: "user", Content: json.RawMessage(`"synthetic retained input"`)}},
		Output: []responseapi.OutputItem{{Type: "message", Role: "assistant", Content: []responseapi.ContentPart{{Type: "output_text", Text: "synthetic retained output"}}}},
	}
	if storeErr := storage.StoreResponse(context.Background(), stored); storeErr != nil {
		t.Fatal(storeErr)
	}
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation).RouterLearning
	router.ResponseAPIFilter = NewResponseAPIFilter(storage)
	router.Config.ModelConfig["cheap"] = addTestQuality(router.Config.ModelConfig[primary], .1)
	router.Config.ModelConfig[primary] = addTestQuality(router.Config.ModelConfig[primary], .98)
	ctx := routerLearningRequestContext("materialized", "conversation")
	ctx.SourceFormat, ctx.TraceContext = llmprotocol.OpenAIResponsesV1, context.Background()
	request, early := router.prepareProtocolRequest([]byte(`{"model":"public","previous_response_id":"resp_synthetic_parent","input":"synthetic next input","store":false}`), ctx)
	if early != nil || request == nil {
		t.Fatal("stored parent could not be prepared")
	}
	if portable, _ := nonPortableContextBinding(ctx); !portable {
		t.Fatal("history must not be considered materialized before expansion")
	}
	if _, snapshotErr := router.extractRequestSignalSnapshot(ctx); snapshotErr != nil {
		t.Fatal(snapshotErr)
	}
	if !ctx.ResponseObjectState.ProviderContextApplied || request.PreviousResponseID != "" || len(request.Messages) != 3 || ctx.PreviousResponseID != stored.ID {
		t.Fatal("history was not expanded before selection, or public lineage was lost")
	}
	if bound, reason := nonPortableContextBinding(ctx); bound {
		t.Fatalf("materialized Router-owned history still treated as opaque: %s", reason)
	}
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: "materialized/conversation", SelectedModel: "cheap", Timestamp: time.Now()})
	refs := []config.ModelRef{{Model: "cheap"}, {Model: primary}}
	decision := &config.Decision{Name: "synthetic-portable", ModelRefs: refs, Algorithm: &config.AlgorithmConfig{
		Type:        config.DecisionAlgorithmMultiFactor,
		MultiFactor: &config.MultiFactorSelectionConfig{Weights: &config.MultiFactorWeightsConfig{Quality: 1}, Quality: &config.QualityEvidenceConfig{Index: testIntelligenceIndex}},
	}}
	ctx.VSRSelectedDecision = decision
	for i := 0; i < 3; i++ {
		router.routerLearningRuntimeState().recordModelExperience(decision.Name, 0, "cheap", routerLearningOutcomeUnderpowered, 1)
		router.routerLearningRuntimeState().recordModelExperience(decision.Name, 0, primary, routerLearningOutcomeGoodFit, 1)
	}
	selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: refs, DecisionName: decision.Name}, decision.Algorithm, ctx)
	if err != nil || selected == nil || selected.Model != primary || ctx.VSRLearningPolicy.Action != routerLearningActionRescueSwitch {
		t.Fatalf("portable retained-history rescue failed: selected=%+v err=%v", selected, err)
	}
	// Materialization proves context portability, not permission to transfer
	// an active tool loop to a different model.
	ctx.VSRConversationFacts = classification.ConversationFacts{LastAssistantToolCall: true}
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: "materialized/conversation", SelectedModel: "cheap", Timestamp: time.Now()})
	selected, _, err = router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: refs[1:], DecisionName: decision.Name}, nil, ctx)
	if selected != nil || !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("materialization bypassed active tool ownership: selected=%+v err=%v", selected, err)
	}
	// A stale or unrelated proof must not make an unresolved identifier portable.
	ctx.ResponseObjectState.PreviousResponseID = "resp_unrelated"
	if bound, _ := nonPortableContextBinding(ctx); !bound {
		t.Fatal("unrelated materialization proof bypassed an opaque identifier")
	}
	unknown := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: context.Background()}
	request, early = router.prepareProtocolRequest([]byte(`{"model":"public","previous_response_id":"resp_synthetic_missing","input":"synthetic next input"}`), unknown)
	if request != nil || early == nil || early.GetImmediateResponse().Status.Code != 404 || unknown.ResponseObjectState != nil {
		t.Fatal("unknown public response ID must fail before materialization or selection")
	}
}

func TestSessionHardBoundaryConflictsRejectBeforeChangingOwnership(t *testing.T) {
	for _, stage := range []string{"initial", "survivors"} {
		for _, boundary := range []string{"tool_loop", "opaque"} {
			for _, mode := range []string{"apply", "observe", "missing_identity", "disabled"} {
				t.Run(stage+"/"+boundary+"/"+mode, func(t *testing.T) {
					sessiontelemetry.ResetRouterSessionMemoryForTesting()
					t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
					router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
					router.Config.RouterLearning = routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation).RouterLearning
					router.Config.ModelConfig[primary] = addTestQuality(router.Config.ModelConfig[primary], .98)
					router.Config.ModelConfig["prior"] = addTestQuality(router.Config.ModelConfig[primary], .1)
					refs := []config.ModelRef{{Model: primary}}
					decision := &config.Decision{Name: "synthetic-conflict"}
					if stage == "survivors" {
						refs = append(refs, config.ModelRef{Model: "prior"})
						decision.Algorithm = &config.AlgorithmConfig{
							Type: config.DecisionAlgorithmMultiFactor,
							MultiFactor: &config.MultiFactorSelectionConfig{
								Quality: &config.QualityEvidenceConfig{Index: testIntelligenceIndex},
								Objective: &config.MultiFactorObjectiveConfig{
									Strategy:   config.MultiFactorObjectiveLexicographic,
									Priorities: []config.MultiFactorPriorityConfig{{Factor: config.MultiFactorFactorQuality, Tolerance: .05}},
								},
							},
						}
					}
					decision.ModelRefs = refs
					ctx := routerLearningRequestContext("conflict", "conversation")
					ctx.VSRSelectedDecision = decision
					if boundary == "tool_loop" {
						ctx.VSRConversationFacts = classification.ConversationFacts{LastAssistantToolCall: true}
					} else {
						ctx.PreviousResponseID = "opaque-unresolved"
					}
					switch mode {
					case "observe":
						decision.Adaptations.Mode = config.DecisionAdaptationModeObserve
					case "missing_identity":
						delete(ctx.Headers, "x-conversation-id")
					case "disabled":
						router.Config.RouterLearning.Protection.Enabled = extprocBoolPtr(false)
					}
					sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: "conflict/conversation", SelectedModel: "prior", Timestamp: time.Now()})
					before, _ := sessiontelemetry.GetRouterSessionSnapshot("conflict/conversation", time.Now())
					selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{CandidateModels: refs, DecisionName: decision.Name}, decision.Algorithm, ctx)
					if mode != "apply" {
						if err != nil || selected == nil || selected.Model != primary {
							t.Fatalf("compatibility mode changed: model=%+v err=%v", selected, err)
						}
						return
					}
					if !errors.Is(err, selection.ErrNoEligibleCandidates) || selected != nil {
						t.Fatalf("hard ownership was silently transferred: model=%+v err=%v", selected, err)
					}
					after, _ := sessiontelemetry.GetRouterSessionSnapshot("conflict/conversation", time.Now())
					if after.CurrentModel != before.CurrentModel || after.TurnCount != before.TurnCount {
						t.Fatal("rejected selection changed session ownership")
					}
					response := router.respondSelectionRejected(ctx, "public", err).GetImmediateResponse()
					if response.Status.Code != 503 {
						t.Fatalf("typed rejection lost terminal HTTP503: %+v", response.Status)
					}
				})
			}
		}
	}
}
