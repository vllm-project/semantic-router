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

func TestSelectModelFromCandidatesFallsBackOnInvalidSelectionResult(t *testing.T) {
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
				t.Fatalf("expected fallback model-a, got %#v", selected)
			}
			if method != string(selection.MethodStatic) {
				t.Fatalf("expected static method, got %q", method)
			}
		})
	}
}

func TestSelectModelFromCandidatesFallsBackToFirstValidCandidateOnInvalidContext(t *testing.T) {
	router := &OpenAIRouter{}
	selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
		CandidateModels: []config.ModelRef{{Model: " "}, {Model: "model-b"}},
	}, nil, nil)

	if selected == nil || selected.Model != "model-b" {
		t.Fatalf("expected fallback model-b, got %#v", selected)
	}
	if method != "" {
		t.Fatalf("expected empty method for invalid context fallback, got %q", method)
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

func TestSelectModelFromCandidatesUsesDecisionScopedSessionAwareConfig(t *testing.T) {
	registry := selection.NewRegistry()
	registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
		SelectedModel: "frontier",
		Score:         0.90,
		Method:        selection.MethodStatic,
		AllScores: map[string]float64{
			"current":  0.20,
			"frontier": 0.90,
		},
	}})

	router := &OpenAIRouter{
		ModelSelector: registry,
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ModelSelection: config.ModelSelectionConfig{
					SessionAware: config.SessionAwareSelectionConfig{
						IdleTimeoutSeconds: extprocIntPtr(300),
					},
				},
			},
		},
	}
	ctx := &RequestContext{SessionID: "decision-session"}
	trueValue := true
	selected, method := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "decision-session",
		DecisionName:    "idle-route",
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
		AgenticSession: &selection.AgenticSessionContext{
			ID:            "decision-session",
			TurnIndex:     3,
			PreviousModel: "current",
			IdleKnown:     true,
			IdleFor:       2 * time.Second,
		},
	}, &config.AlgorithmConfig{
		Type: "session_aware",
		SessionAware: &config.SessionAwareSelectionConfig{
			BaseMethod:           "static",
			IdleTimeoutSeconds:   extprocIntPtr(1),
			MinTurnsBeforeSwitch: extprocIntPtr(1),
			SwitchMargin:         extprocFloat64Ptr(0.05),
			StayBias:             extprocFloat64Ptr(0.10),
			ToolLoopHardLock:     &trueValue,
		},
	}, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected frontier from decision-scoped idle policy, got %#v", selected)
	}
	if method != string(selection.MethodSessionAware) {
		t.Fatalf("expected session_aware method, got %q", method)
	}
	if ctx.VSRSessionPolicy == nil || ctx.VSRSessionPolicy["idle_expired"] != true {
		t.Fatalf("expected decision-scoped idle timeout in session policy, got %#v", ctx.VSRSessionPolicy)
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

func TestApplySessionAwareSelectionConfigPreservesExplicitZeroValues(t *testing.T) {
	cfg := selection.DefaultSessionAwareConfig()
	falseValue := false
	applySessionAwareSelectionConfig(cfg, config.SessionAwareSelectionConfig{
		IdleTimeoutSeconds:         extprocIntPtr(0),
		MinTurnsBeforeSwitch:       extprocIntPtr(0),
		SwitchMargin:               extprocFloat64Ptr(0),
		StayBias:                   extprocFloat64Ptr(0),
		ToolLoopHardLock:           &falseValue,
		ContextPortabilityHardLock: &falseValue,
		DecisionDriftReset:         &falseValue,
		PrefixCacheWeight:          extprocFloat64Ptr(0),
		RemainingTurnPriorWeight:   extprocFloat64Ptr(0),
	})

	if cfg.IdleTimeoutSeconds != 0 ||
		cfg.MinTurnsBeforeSwitch != 0 ||
		cfg.SwitchMargin != 0 ||
		cfg.StayBias != 0 ||
		cfg.ToolLoopHardLock ||
		cfg.ContextPortabilityHardLock ||
		cfg.DecisionDriftReset ||
		cfg.PrefixCacheWeight != 0 ||
		cfg.RemainingTurnPriorWeight != 0 {
		t.Fatalf("expected explicit zero/false session-aware config to be preserved, got %#v", cfg)
	}
}

func extprocIntPtr(v int) *int {
	return &v
}

func extprocFloat64Ptr(v float64) *float64 {
	return &v
}
