/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

type selectionResultSelector struct {
	result *selection.SelectionResult
	err    error
}

func (s selectionResultSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	return s.result, s.err
}

func (s selectionResultSelector) Method() selection.SelectionMethod {
	return selection.MethodStatic
}

func (s selectionResultSelector) UpdateFeedback(ctx context.Context, feedback *selection.Feedback) error {
	return nil
}

func (s selectionResultSelector) Tier() selection.AlgorithmTier {
	return selection.TierSupported
}

func (s selectionResultSelector) ExternalDependencies() []selection.Dependency {
	return nil
}

func TestSelectModelFromCandidatesUsesDefaultCandidateOnInvalidSelectionResult(t *testing.T) {
	for _, tc := range []struct {
		name   string
		result *selection.SelectionResult
	}{
		{
			name: "nil result",
		},
		{
			name:   "non candidate result",
			result: &selection.SelectionResult{SelectedModel: "model-c"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			registry := selection.NewRegistry()
			registry.Register(selection.MethodStatic, selectionResultSelector{result: tc.result})

			router := &OpenAIRouter{ModelSelector: registry}
			selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
				CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
			}, nil, nil)

			if selected == nil || selected.Model != "model-a" {
				t.Fatalf("expected default candidate model-a, got %#v", selected)
			}
			if method != string(selection.MethodStatic) {
				t.Fatalf("expected static method, got %q", method)
			}
		})
	}
}

func TestSelectModelFromCandidatesUsesFirstValidDefaultCandidateOnInvalidContext(t *testing.T) {
	router := &OpenAIRouter{}
	selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
		CandidateModels: []config.ModelRef{{Model: " "}, {Model: "model-b"}},
	}, nil, nil)

	if selected == nil || selected.Model != "model-b" {
		t.Fatalf("expected default candidate model-b, got %#v", selected)
	}
	if method != "" {
		t.Fatalf("expected empty method for invalid context default, got %q", method)
	}
}

func TestSelectModelFromCandidatesRecordsSingleCandidateInRouterMemory(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{}
	reqCtx := &RequestContext{SessionID: "single-candidate-session"}
	selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "single-candidate-session",
		DecisionName:    "warmup",
		CandidateModels: []config.ModelRef{{Model: "model-a"}},
	}, nil, reqCtx)

	if selected == nil || selected.Model != "model-a" {
		t.Fatalf("expected model-a, got %#v", selected)
	}
	if method != "single" {
		t.Fatalf("expected single method, got %q", method)
	}

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("single-candidate-session", time.Now())
	if !ok {
		t.Fatal("expected router memory snapshot for single-candidate selection")
	}
	if snapshot.CurrentModel != "model-a" {
		t.Fatalf("expected current model model-a, got %q", snapshot.CurrentModel)
	}
}

func TestRouterLearningSessionAwareKeepsCurrentModelAcrossDecisionCandidates(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:      "session-a/conversation-a",
		SelectedModel:  "frontier",
		DecisionName:   "complex-code",
		TurnIndex:      2,
		ActiveToolLoop: true,
		Timestamp:      time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{Name: "simple-followup"}
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}

	selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "simple-followup",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected learning to keep frontier across decision candidates, got %#v", selected)
	}
	if method != "single" {
		t.Fatalf("expected base method single, got %q", method)
	}
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionHardLock) {
		t.Fatalf("expected hard_lock learning action, got %#v", ctx.VSRLearningPolicy)
	}
	policyMap := ctx.VSRLearningPolicy.ToMap()
	if _, ok := policyMap["session_id"]; ok {
		t.Fatalf("learning diagnostics must not expose raw session_id: %#v", ctx.VSRLearningPolicy)
	}
	if _, ok := policyMap["conversation_id"]; ok {
		t.Fatalf("learning diagnostics must not expose raw conversation_id: %#v", ctx.VSRLearningPolicy)
	}
	sessionIdentity := learningIdentityPart(t, ctx.VSRLearningPolicy, "session")
	if sessionIdentity["status"] != "present" || sessionIdentity["source"] != "header:x-session-id" {
		t.Fatalf("expected hashed session identity diagnostics, got %#v", sessionIdentity)
	}
	if sessionIdentity["hash"] == "session-a" || sessionIdentity["hash"] == "" {
		t.Fatalf("expected non-raw session identity hash, got %#v", sessionIdentity)
	}
	if ctx.VSRLearningSessionID != "session-a/conversation-a" {
		t.Fatalf("expected conversation memory key, got %q", ctx.VSRLearningSessionID)
	}
}

func TestRouterLearningSessionAwareReleasesOnNewConversationScope(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "session-a/conversation-a",
		SelectedModel: "frontier",
		DecisionName:  "complex-code",
		TurnIndex:     2,
		Timestamp:     time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-b")
	ctx.VSRSelectedDecision = &config.Decision{Name: "simple-new-run"}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "simple-new-run",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "cheap" {
		t.Fatalf("expected new conversation to route to cheap base candidate, got %#v", selected)
	}
	if ctx.VSRLearningSessionID != "session-a/conversation-b" {
		t.Fatalf("expected new conversation memory key, got %q", ctx.VSRLearningSessionID)
	}
}

func TestRouterLearningSessionAwareSessionScopeProtectsAcrossConversations(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "session-a",
		SelectedModel: "frontier",
		DecisionName:  "complex-code",
		TurnIndex:     2,
		Timestamp:     time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeSession)}
	ctx := routerLearningRequestContext("session-a", "conversation-b")
	ctx.VSRSelectedDecision = &config.Decision{Name: "simple-new-run"}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "simple-new-run",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected session scope to keep frontier, got %#v", selected)
	}
	if ctx.VSRLearningSessionID != "session-a" {
		t.Fatalf("expected session memory key, got %q", ctx.VSRLearningSessionID)
	}
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionStay) {
		t.Fatalf("expected session scope stay action, got %#v", ctx.VSRLearningPolicy)
	}
	if ctx.VSRLearningPolicy.String("reason") != "session_scope_protect" {
		t.Fatalf("expected session scope protect reason, got %#v", ctx.VSRLearningPolicy)
	}
	conversationIdentity := learningIdentityPart(t, ctx.VSRLearningPolicy, "conversation")
	if conversationIdentity["status"] != "present" || conversationIdentity["required"] != false {
		t.Fatalf("expected optional conversation identity for session scope, got %#v", conversationIdentity)
	}
}

func TestRouterLearningDecisionScopeOverrideProtectsAcrossConversations(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "session-a",
		SelectedModel: "frontier",
		DecisionName:  "complex-code",
		TurnIndex:     2,
		Timestamp:     time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-b")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "session-sticky",
		Adaptations: config.DecisionAdaptationsConfig{
			SessionAware: &config.DecisionSessionAwareAdaptationConfig{
				Scope: config.RouterLearningScopeSession,
			},
		},
	}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "session-sticky",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected decision scope override to keep frontier, got %#v", selected)
	}
	if ctx.VSRLearningSessionID != "session-a" {
		t.Fatalf("expected session memory key from decision override, got %q", ctx.VSRLearningSessionID)
	}
	if ctx.VSRLearningPolicy.String("scope") != config.RouterLearningScopeSession {
		t.Fatalf("expected session-scope learning policy, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestRouterLearningSessionAwareObserveRecordsWithoutChangingModel(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:      "session-a/conversation-a",
		SelectedModel:  "frontier",
		DecisionName:   "complex-code",
		TurnIndex:      2,
		ActiveToolLoop: true,
		Timestamp:      time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "observe-route",
		Adaptations: config.DecisionAdaptationsConfig{
			SessionAware: &config.DecisionSessionAwareAdaptationConfig{
				Mode: config.DecisionAdaptationModeObserve,
			},
		},
	}
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "observe-route",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "cheap" {
		t.Fatalf("expected observe mode to keep base candidate, got %#v", selected)
	}
	if ctx.VSRLearningPolicy.String("mode") != config.DecisionAdaptationModeObserve ||
		ctx.VSRLearningPolicy.String("action") != string(routerLearningActionHardLock) {
		t.Fatalf("expected observe hard_lock policy, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestRouterLearningSessionAwareMissingIdentityNoOps(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := &RequestContext{
		Headers: map[string]string{"x-session-id": "session-a"},
	}
	ctx.VSRSelectedDecision = &config.Decision{Name: "simple"}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "simple",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "cheap" {
		t.Fatalf("expected missing conversation identity to keep base candidate, got %#v", selected)
	}
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionNoop) ||
		ctx.VSRLearningPolicy.String("reason") != "identity_missing" {
		t.Fatalf("expected identity_missing noop policy, got %#v", ctx.VSRLearningPolicy)
	}
	conversationIdentity := learningIdentityPart(t, ctx.VSRLearningPolicy, "conversation")
	if conversationIdentity["status"] != "missing" || conversationIdentity["required"] != true {
		t.Fatalf("expected missing required conversation identity diagnostics, got %#v", conversationIdentity)
	}
}

func TestRouterLearningSessionAwareBypassLeavesBaseDecisionFinal(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:      "session-a/conversation-a",
		SelectedModel:  "frontier",
		DecisionName:   "complex-code",
		TurnIndex:      2,
		ActiveToolLoop: true,
		Timestamp:      time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "privacy",
		Adaptations: config.DecisionAdaptationsConfig{
			SessionAware: &config.DecisionSessionAwareAdaptationConfig{Mode: config.DecisionAdaptationModeBypass},
		},
	}
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "privacy",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "cheap" {
		t.Fatalf("expected bypass decision to keep base candidate, got %#v", selected)
	}
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionBypass) {
		t.Fatalf("expected bypass learning policy, got %#v", ctx.VSRLearningPolicy)
	}
	if ctx.VSRLearningPolicy.String("scope") != config.RouterLearningScopeConversation {
		t.Fatalf("expected bypass learning policy to include scope, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestSelectorForDecisionMethodBuildsDecisionScopedHybridSelector(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.BackendModels.ModelConfig = map[string]config.ModelParams{
		"current":  {Description: "general chat"},
		"frontier": {Description: "coding specialist"},
	}

	modelSelectionCfg := buildModelSelectionConfig(&cfg)
	registry := selection.NewFactory(modelSelectionCfg).
		WithModelConfig(cfg.BackendModels.ModelConfig).
		WithEmbeddingFunc(func(text string) ([]float32, error) {
			lower := strings.ToLower(text)
			switch {
			case strings.Contains(lower, "coding"):
				return []float32{1, 0}, nil
			case strings.Contains(lower, "general"):
				return []float32{0, 1}, nil
			default:
				return []float32{0.5, 0.5}, nil
			}
		}).
		CreateAll()

	router := &OpenAIRouter{
		Config:        &cfg,
		ModelSelector: registry,
	}

	selector := router.selectorForDecisionMethod(selection.MethodHybrid, &config.AlgorithmConfig{
		Type: "hybrid",
		Hybrid: &config.HybridSelectionConfig{
			EloWeight:      0.6,
			RouterDCWeight: 0.4,
		},
	})

	result, err := selector.Select(context.Background(), &selection.SelectionContext{
		Query:           "need help with coding",
		DecisionName:    "hybrid_route",
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	wantWeights := fmt.Sprintf("weights=[elo:%.2f, dc:%.2f, am:%.2f, cost:%.2f]", 0.6, 0.4, 0.2, 0.2)
	if !strings.Contains(result.Reasoning, wantWeights) {
		t.Fatalf("expected decision-scoped hybrid weights in reasoning, got %q", result.Reasoning)
	}
}

func TestBuildSelectionContextUsesPinnedSessionIDAndToolLoopFacts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {ContextWindowSize: 8192},
			},
		},
	}}
	reqCtx := &RequestContext{
		SessionID:            "pinned-session",
		PreviousModel:        "model-a",
		TurnIndex:            2,
		HistoryTokenCount:    1024,
		VSRContextTokenCount: 2048,
		SessionIdleSeconds:   12,
		SessionIdleKnown:     true,
		VSRConversationFacts: classification.ConversationFacts{
			AssistantToolCallCount: 1,
			ToolResultCount:        1,
			LastMessageRole:        "tool",
			LastMessageToolResult:  true,
		},
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}},
		"agentic",
		"query",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.SessionID != "pinned-session" {
		t.Fatalf("expected pinned session ID, got %q", selCtx.SessionID)
	}
	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.ActiveToolLoop {
		t.Fatalf("expected active tool loop in agentic session context: %#v", selCtx.AgenticSession)
	}
	if got := selCtx.AgenticSession.ModelContextWindows["model-a"]; got != 8192 {
		t.Fatalf("expected model context window 8192, got %d", got)
	}
}

func TestBuildSelectionContextMarksUserAfterToolResultAsToolLoop(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	reqCtx := &RequestContext{
		SessionID:     "tool-continuation-session",
		PreviousModel: "model-a",
		VSRConversationFacts: classification.ConversationFacts{
			AssistantToolCallCount:  1,
			ToolResultCount:         1,
			LastMessageRole:         "user",
			LastUserAfterToolResult: true,
		},
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		"agentic",
		"continue after tool output",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.ActiveToolLoop {
		t.Fatalf("expected user-after-tool continuation to be an active tool loop: %#v", selCtx.AgenticSession)
	}
	if selCtx.AgenticSession.Phase != selection.AgenticPhaseToolLoop {
		t.Fatalf("expected tool-loop phase, got %q", selCtx.AgenticSession.Phase)
	}
}

func TestBuildSelectionContextMarksPreviousResponseIDAsNonPortableContext(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	reqCtx := &RequestContext{
		SessionID:          "response-api-session",
		PreviousModel:      "model-a",
		PreviousResponseID: "resp_123",
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		"agentic",
		"continue response",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.HasNonPortableContext {
		t.Fatalf("expected previous_response_id to mark non-portable context: %#v", selCtx.AgenticSession)
	}
	if got := selCtx.AgenticSession.NonPortableContextReason; got != "previous_response_id" {
		t.Fatalf("expected previous_response_id reason, got %q", got)
	}
	if got := selCtx.AgenticSession.Phase; got != selection.AgenticPhaseProviderState {
		t.Fatalf("expected provider-state phase for previous_response_id, got %q", got)
	}
}

func TestSessionAwareLearningConfigPreservesExplicitZeroValues(t *testing.T) {
	cfg := sessionAwareSelectionConfigFromLearning(config.SessionAwareLearningConfig{
		Tuning: config.SessionAwareLearningTuning{
			IdleTimeoutSeconds:     extprocIntPtr(0),
			MinTurnsBeforeSwitch:   extprocIntPtr(0),
			SwitchMargin:           extprocFloat64Ptr(0),
			CacheWeight:            extprocFloat64Ptr(0),
			HandoffPenalty:         extprocFloat64Ptr(0),
			HandoffPenaltyWeight:   extprocFloat64Ptr(0),
			SwitchHistoryWeight:    extprocFloat64Ptr(0),
			MaxCacheCostMultiplier: extprocFloat64Ptr(0),
		},
	})

	if cfg.IdleTimeoutSeconds != 0 ||
		cfg.MinTurnsBeforeSwitch != 0 ||
		cfg.SwitchMargin != 0 ||
		cfg.PrefixCacheWeight != 0 ||
		cfg.DefaultHandoffPenalty != 0 ||
		cfg.HandoffPenaltyWeight != 0 ||
		cfg.SwitchHistoryWeight != 0 ||
		cfg.MaxCacheCostMultiplier != 0 {
		t.Fatalf("expected explicit zero session-aware learning tuning to be preserved, got %#v", cfg)
	}
}

func extprocIntPtr(v int) *int {
	return &v
}

func extprocFloat64Ptr(v float64) *float64 {
	return &v
}

func routerLearningTestConfig(scope string) *config.RouterConfig {
	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "cheap",
			ModelConfig: map[string]config.ModelParams{
				"cheap":    {},
				"frontier": {},
			},
		},
		RouterLearning: config.RouterLearningConfig{
			Enabled: true,
			Adaptations: config.RouterLearningAdaptations{
				SessionAware: config.SessionAwareLearningConfig{
					Enabled: true,
					Scope:   scope,
					Identity: config.SessionAwareIdentityConfig{
						Headers: map[string]string{
							"session":      "x-session-id",
							"conversation": "x-conversation-id",
						},
					},
				},
			},
		},
	}
}

func routerLearningRequestContext(sessionID string, conversationID string) *RequestContext {
	return &RequestContext{
		Headers: map[string]string{
			"x-session-id":      sessionID,
			"x-conversation-id": conversationID,
		},
		SessionID: sessionID,
	}
}

func learningIdentityPart(t *testing.T, policy *routerLearningPolicy, name string) map[string]interface{} {
	t.Helper()
	policyMap := policy.ToMap()
	identity, ok := policyMap["identity"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected identity diagnostics in learning policy, got %#v", policy)
	}
	part, ok := identity[name].(map[string]interface{})
	if !ok {
		t.Fatalf("expected %s identity diagnostics, got %#v", name, identity)
	}
	return part
}
