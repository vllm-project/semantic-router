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
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestRouterLearningProtectionKeepsCurrentModelAcrossDecisionCandidates(t *testing.T) {
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
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionHoldCurrent) {
		t.Fatalf("expected hold_current learning action, got %#v", ctx.VSRLearningPolicy)
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

func TestRouterLearningProtectionReleasesOnNewConversationScope(t *testing.T) {
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
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionEstablishBaseline) ||
		ctx.VSRLearningPolicy.String("reason") != "new_conversation" {
		t.Fatalf("expected establish_baseline policy for new conversation, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestRouterLearningProtectionSessionScopeProtectsAcrossConversations(t *testing.T) {
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
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionHoldCurrent) {
		t.Fatalf("expected session scope hold_current action, got %#v", ctx.VSRLearningPolicy)
	}
	if ctx.VSRLearningPolicy.String("reason") != "session_scope_protect" {
		t.Fatalf("expected session scope protect reason, got %#v", ctx.VSRLearningPolicy)
	}
	conversationIdentity := learningIdentityPart(t, ctx.VSRLearningPolicy, "conversation")
	if conversationIdentity["status"] != "present" || conversationIdentity["required"] != false {
		t.Fatalf("expected optional conversation identity for session scope, got %#v", conversationIdentity)
	}
}

func TestRouterLearningProtectionRescueSwitchEscapesUnderpoweredCurrentModel(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "session-a",
		SelectedModel: "cheap",
		DecisionName:  "simple",
		TurnIndex:     3,
		Timestamp:     time.Now(),
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeSession)}
	for i := 0; i < 3; i++ {
		router.routerLearningRuntimeState().recordModelExperience(
			"complex-rescue",
			3,
			"cheap",
			routerLearningOutcomeUnderpowered,
			1,
		)
		router.routerLearningRuntimeState().recordModelExperience(
			"complex-rescue",
			3,
			"frontier",
			routerLearningOutcomeGoodFit,
			1,
		)
	}
	ctx := routerLearningRequestContext("session-a", "conversation-b")
	ctx.VSRSelectedDecision = &config.Decision{Name: "complex-rescue", Tier: 3}
	selCtx := &selection.SelectionContext{
		SessionID:    "session-a",
		DecisionName: "complex-rescue",
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
	}
	baseResult := &selection.SelectionResult{
		SelectedModel: "frontier",
		Score:         0.9,
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		AllScores: map[string]float64{
			"cheap":    0.1,
			"frontier": 0.9,
		},
	}

	_, result, selected, _ := router.applyRouterLearning(selCtx, baseResult, &selCtx.CandidateModels[1], ctx)

	if selected == nil || selected.Model != "frontier" || result.SelectedModel != "frontier" {
		t.Fatalf("expected rescue switch to frontier, got result=%#v selected=%#v", result, selected)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok {
		t.Fatalf("expected protection policy, got %#v", ctx.VSRLearningPolicies)
	}
	if policy.Action != routerLearningActionRescueSwitch || policy.Reason != "rescue_evidence" {
		t.Fatalf("expected rescue_switch protection policy, got %#v", policy.ToMap())
	}
	replay := policy.toReplayProtection()
	if replay == nil || replay.Rescue == nil || !replay.Rescue.Active {
		t.Fatalf("expected replay rescue diagnostics, got %#v", replay)
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

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeSession)}
	ctx := routerLearningRequestContext("session-a", "conversation-b")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "session-sticky",
	}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "session-sticky",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected session-scope protection to keep frontier, got %#v", selected)
	}
	if ctx.VSRLearningSessionID != "session-a" {
		t.Fatalf("expected session memory key, got %q", ctx.VSRLearningSessionID)
	}
	if ctx.VSRLearningPolicy.String("scope") != config.RouterLearningScopeSession {
		t.Fatalf("expected session-scope learning policy, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestRouterLearningProtectionObserveRecordsWithoutChangingModel(t *testing.T) {
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
			Protection: &config.DecisionLearningProtectionConfig{Mode: config.DecisionAdaptationModeObserve},
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
		ctx.VSRLearningPolicy.String("action") != string(routerLearningActionObserve) ||
		ctx.VSRLearningPolicy.String("reason") != "observe_only" {
		t.Fatalf("expected observe-only policy, got %#v", ctx.VSRLearningPolicy)
	}
}

func TestRouterLearningProtectionObserveDoesNotRollbackAdaptationProposal(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() {
		routerLearningSamplingSeedSource = originalSeedSource
	})

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		50,
	)
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"cheap",
		routerLearningOutcomeUnderpowered,
		50,
	)
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "adaptive",
		Tier: 2,
		Adaptations: config.DecisionAdaptationsConfig{
			Protection: &config.DecisionLearningProtectionConfig{Mode: config.DecisionAdaptationModeObserve},
		},
	}
	selCtx := &selection.SelectionContext{
		SessionID:    "session-a",
		DecisionName: "adaptive",
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
	}
	baseResult := &selection.SelectionResult{
		SelectedModel: "cheap",
		Score:         1,
		Method:        selection.MethodStatic,
		AllScores:     map[string]float64{"cheap": 1},
	}

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, &selCtx.CandidateModels[0], ctx)

	if !applied || selected == nil || selected.Model != "frontier" || result.SelectedModel != "frontier" {
		t.Fatalf("expected protection observe to preserve adaptation proposal, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok {
		t.Fatalf("expected protection policy, got %#v", ctx.VSRLearningPolicies)
	}
	if policy.Action != routerLearningActionObserve ||
		policy.Reason != "observe_only" ||
		policy.String("proposal_model") != "frontier" ||
		policy.String("final_model") != "frontier" {
		t.Fatalf("expected protection observe policy to report frontier proposal/final, got %#v", policy.ToMap())
	}
}

func TestRouterLearningProtectionMissingIdentityNoOps(t *testing.T) {
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
	if ctx.VSRLearningPolicy.String("action") != string(routerLearningActionSuppressSampling) ||
		ctx.VSRLearningPolicy.String("reason") != "missing_identity" {
		t.Fatalf("expected missing_identity fail-open policy, got %#v", ctx.VSRLearningPolicy)
	}
	conversationIdentity := learningIdentityPart(t, ctx.VSRLearningPolicy, "conversation")
	if conversationIdentity["status"] != "missing" || conversationIdentity["required"] != true {
		t.Fatalf("expected missing required conversation identity diagnostics, got %#v", conversationIdentity)
	}
}

func TestRouterLearningProtectionBypassLeavesBaseDecisionFinal(t *testing.T) {
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
		Name:        "privacy",
		Adaptations: config.DecisionAdaptationsConfig{Mode: config.DecisionAdaptationModeBypass},
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

func TestRouterLearningDecisionObserveModeRecordsProposalWithoutChangingModel(t *testing.T) {
	router := &OpenAIRouter{Config: routerLearningAdaptationTestConfig()}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		10,
	)
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "adaptive",
			Tier: 2,
			Adaptations: config.DecisionAdaptationsConfig{
				Mode: config.DecisionAdaptationModeObserve,
			},
		},
	}
	selCtx := &selection.SelectionContext{
		DecisionName: "adaptive",
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
	}
	baseResult := &selection.SelectionResult{
		SelectedModel: "cheap",
		Score:         1,
		Method:        selection.MethodStatic,
		AllScores:     map[string]float64{"cheap": 1},
	}

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, &selCtx.CandidateModels[0], ctx)

	if applied {
		t.Fatal("expected observe mode to leave final model unchanged")
	}
	if selected == nil || selected.Model != "cheap" || result.SelectedModel != "cheap" {
		t.Fatalf("expected observe mode to keep base cheap model, got result=%#v selected=%#v", result, selected)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
	if !ok {
		t.Fatalf("expected adaptation policy, got %#v", ctx.VSRLearningPolicies)
	}
	if policy.String("action") != string(routerLearningActionObserve) ||
		policy.String("proposal_model") != "frontier" {
		t.Fatalf("expected adaptation observe policy with frontier proposal, got %#v", policy.ToMap())
	}
}

func TestRouterLearningAdaptationUsesSamplingOnlyWhenPreflightAllows(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() {
		routerLearningSamplingSeedSource = originalSeedSource
	})

	router := &OpenAIRouter{Config: routerLearningAdaptationTestConfig()}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		50,
	)
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"cheap",
		routerLearningOutcomeUnderpowered,
		50,
	)
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "adaptive",
			Tier: 2,
		},
	}
	selCtx := &selection.SelectionContext{
		DecisionName: "adaptive",
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
	}
	baseResult := &selection.SelectionResult{
		SelectedModel: "cheap",
		Score:         1,
		Method:        selection.MethodStatic,
		AllScores:     map[string]float64{"cheap": 1},
	}
	input := routerLearningInput{
		selCtx:           selCtx,
		baseResult:       baseResult,
		selectedModelRef: &selCtx.CandidateModels[0],
		ctx:              ctx,
	}
	cfg, ok := router.adaptationConfig(selCtx, baseResult, &selCtx.CandidateModels[0], ctx)
	if !ok {
		t.Fatal("expected adaptation config")
	}

	allowed := router.applyRoutingSamplingAdaptation(input, routerLearningProtectionPreflight{
		enabled:         true,
		samplingAllowed: true,
	}, cfg)
	if allowed.policy.Action != routerLearningActionProposeSwitch {
		t.Fatalf("expected switch proposal when preflight allows sampling, got %#v", allowed.policy.ToMap())
	}
	allowedSampling, ok := allowed.policy.ToMap()["sampling"].(map[string]interface{})
	if !ok || allowedSampling["used"] != true {
		t.Fatalf("expected sampling diagnostics to mark used=true, got %#v", allowed.policy.ToMap())
	}
	if allowedSampling["seed"] != int64(424242) {
		t.Fatalf("expected injectable sampling seed in diagnostics, got %#v", allowedSampling)
	}

	suppressed := router.applyRoutingSamplingAdaptation(input, routerLearningProtectionPreflight{
		enabled:         true,
		samplingAllowed: false,
	}, cfg)
	if suppressed.policy.Action != routerLearningActionProposeSwitch {
		t.Fatalf("expected deterministic proposal when sampling is suppressed, got %#v", suppressed.policy.ToMap())
	}
	suppressedSampling, ok := suppressed.policy.ToMap()["sampling"].(map[string]interface{})
	if !ok || suppressedSampling["used"] != false {
		t.Fatalf("expected sampling diagnostics to mark used=false, got %#v", suppressed.policy.ToMap())
	}
	if _, ok := suppressedSampling["seed"]; ok {
		t.Fatalf("did not expect deterministic path to record a sampling seed, got %#v", suppressedSampling)
	}
}

func TestRouterLearningCandidateSetsAreDeterministicAndEligible(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "cheap",
			ModelConfig: map[string]config.ModelParams{
				"cheap":          {},
				"endpoint-only":  {},
				"frontier":       {},
				"global-only":    {},
				"inventory-only": {},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "endpoint-only", Model: "endpoint-only"},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "simple",
					Tier:      2,
					ModelRefs: []config.ModelRef{{Model: "cheap"}, {Model: "missing"}},
				},
				{
					Name:      "complex",
					Tier:      2,
					ModelRefs: []config.ModelRef{{Model: "frontier"}, {Model: "cheap"}},
				},
				{
					Name:      "other",
					Tier:      4,
					ModelRefs: []config.ModelRef{{Model: "global-only"}},
				},
			},
		},
	}}
	ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Name: "simple", Tier: 2}}
	selCtx := &selection.SelectionContext{
		DecisionName:    "simple",
		CandidateModels: []config.ModelRef{{Model: "cheap"}, {Model: "missing"}},
	}

	assertModelRefs(t, router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetDecision), []string{"cheap"})
	assertModelRefs(t, router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetTier), []string{"cheap", "frontier"})
	assertModelRefs(t, router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetGlobal), []string{"cheap", "frontier", "global-only", "endpoint-only", "inventory-only"})
}

func TestRouterLearningAdaptationUsesDecisionCandidateSetOverride(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "cheap",
			ModelConfig: map[string]config.ModelParams{
				"cheap":    {},
				"frontier": {},
			},
		},
		RouterLearning: config.RouterLearningConfig{
			Enabled: true,
			Adaptation: config.RouterLearningAdaptationConfig{
				CandidateSet: config.RouterLearningCandidateSetDecision,
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "simple",
					Tier:      2,
					ModelRefs: []config.ModelRef{{Model: "cheap"}},
					Adaptations: config.DecisionAdaptationsConfig{
						Adaptation: &config.DecisionLearningAdaptationConfig{
							CandidateSet: config.RouterLearningCandidateSetTier,
						},
					},
				},
				{
					Name:      "complex",
					Tier:      2,
					ModelRefs: []config.ModelRef{{Model: "frontier"}},
				},
			},
		},
	}}
	ctx := &RequestContext{VSRSelectedDecision: &router.Config.Decisions[0]}
	selCtx := &selection.SelectionContext{
		DecisionName:    "simple",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}
	baseResult := &selection.SelectionResult{SelectedModel: "cheap"}

	adaptationCfg, ok := router.adaptationConfig(selCtx, baseResult, &selCtx.CandidateModels[0], ctx)
	if !ok {
		t.Fatal("expected adaptation to be enabled")
	}
	if got := adaptationCfg.EffectiveCandidateSet(); got != config.RouterLearningCandidateSetTier {
		t.Fatalf("expected decision candidate_set override, got %q", got)
	}
	learningCtx := router.adaptationSelectionContext(selCtx, ctx, adaptationCfg.EffectiveCandidateSet())
	assertModelRefs(t, learningCtx.CandidateModels, []string{"cheap", "frontier"})
}

func TestRouterLearningProtectionConfigPreservesExplicitZeroValues(t *testing.T) {
	cfg := protectionSelectionConfig(config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			IdleTimeoutSeconds:   extprocIntPtr(0),
			MinTurnsBeforeSwitch: extprocIntPtr(0),
			SwitchMargin:         extprocFloat64Ptr(0),
			StabilityWeight:      extprocFloat64Ptr(0),
		},
	})

	if cfg.IdleTimeoutSeconds != 0 ||
		cfg.MinTurnsBeforeSwitch != 0 ||
		cfg.SwitchMargin != 0 ||
		cfg.PrefixCacheWeight != 0 ||
		cfg.HandoffPenaltyWeight != 0 ||
		cfg.SwitchHistoryWeight != 0 {
		t.Fatalf("expected explicit zero Router Learning protection tuning to be preserved, got %#v", cfg)
	}
}

func TestRouterLearningProtectionConfigAppliesStabilityWeightToSwitchCosts(t *testing.T) {
	cfg := protectionSelectionConfig(config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			SwitchMargin:    extprocFloat64Ptr(0.05),
			StabilityWeight: extprocFloat64Ptr(2),
		},
	})

	if cfg.SwitchMargin != 0.05 {
		t.Fatalf("stability weight must not multiply switch_margin, got %v", cfg.SwitchMargin)
	}
	if cfg.PrefixCacheWeight != 0.4 ||
		cfg.HandoffPenaltyWeight != 2 ||
		cfg.SwitchHistoryWeight != 0.08 ||
		cfg.ToolLoopStayBias != 0.70 {
		t.Fatalf("expected stability weight to scale switch costs, got %#v", cfg)
	}
}
