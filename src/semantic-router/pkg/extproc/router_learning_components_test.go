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

func TestRouterLearningAdaptationOnlyCanSwitchWhenProtectionDisabled(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = originalSeedSource })

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
		VSRSelectedDecision: &config.Decision{Name: "adaptive", Tier: 2},
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

	if !applied || selected == nil || selected.Model != "frontier" || result.SelectedModel != "frontier" {
		t.Fatalf("expected adaptation-only switch to frontier, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	if _, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection); ok {
		t.Fatalf("expected no protection policy when protection is disabled, got %#v", ctx.VSRLearningPolicies)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
	if !ok || policy.Action != routerLearningActionProposeSwitch || policy.Reason != "sampled_win" {
		t.Fatalf("expected sampled adaptation switch proposal, got %#v", ctx.VSRLearningPolicies)
	}
	sampling, ok := policy.ToMap()["sampling"].(map[string]interface{})
	if !ok || sampling["used"] != true || sampling["seed"] != int64(424242) {
		t.Fatalf("expected protection-disabled adaptation to sample, got %#v", policy.ToMap())
	}
}

func TestRouterLearningUnknownAdaptationStrategyKeepsBaseModel(t *testing.T) {
	cfg := routerLearningAdaptationTestConfig()
	cfg.RouterLearning.Adaptation.Strategy = "missing_strategy"
	router := &OpenAIRouter{Config: cfg}
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
		VSRSelectedDecision: &config.Decision{Name: "adaptive", Tier: 2},
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

	if applied || selected == nil || selected.Model != "cheap" || result.SelectedModel != "cheap" {
		t.Fatalf("expected unknown strategy to keep base model, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
	if !ok || policy.Action != routerLearningActionKeepBase || policy.Reason != "strategy_unavailable" {
		t.Fatalf("expected strategy_unavailable keep_base policy, got %#v", ctx.VSRLearningPolicies)
	}
	if _, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection); ok {
		t.Fatalf("expected no protection policy when protection is disabled, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestRouterLearningProtectionOnlyCanGuardWhenAdaptationDisabled(t *testing.T) {
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

	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{Name: "simple-followup"}
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}

	selected, _ := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "session-a",
		DecisionName:    "simple-followup",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}, nil, ctx)

	if selected == nil || selected.Model != "frontier" {
		t.Fatalf("expected protection-only guard to keep frontier, got %#v", selected)
	}
	if _, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation); ok {
		t.Fatalf("expected no adaptation policy when adaptation is disabled, got %#v", ctx.VSRLearningPolicies)
	}
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok || policy.Action != routerLearningActionHoldCurrent {
		t.Fatalf("expected protection hold_current policy, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestRouterLearningProtectionObserveDoesNotSuppressAdaptationSampling(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = originalSeedSource })

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		1000,
	)
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"cheap",
		routerLearningOutcomeUnderpowered,
		1000,
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
		t.Fatalf("expected adaptation to switch while protection observes, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	assertAdaptationSampled(t, ctx, true)
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok || policy.Action != routerLearningActionObserve || policy.Reason != "observe_only" {
		t.Fatalf("expected protection observe policy, got %#v", ctx.VSRLearningPolicies)
	}
	preflight := ctx.VSRLearningProtectionPreflight
	if preflight == nil || preflight.Action != string(routerLearningActionObserve) {
		t.Fatalf("expected protection preflight to be diagnostic observe-only, got %#v", preflight)
	}
}

func TestRouterLearningProtectionBypassDoesNotSuppressAdaptationSampling(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = originalSeedSource })

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		1000,
	)
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"cheap",
		routerLearningOutcomeUnderpowered,
		1000,
	)
	ctx := routerLearningRequestContext("session-a", "conversation-a")
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "adaptive",
		Tier: 2,
		Adaptations: config.DecisionAdaptationsConfig{
			Protection: &config.DecisionLearningProtectionConfig{Mode: config.DecisionAdaptationModeBypass},
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

	if !applied || selected == nil || selected.Model != "frontier" || result.SelectedModel != "frontier" {
		t.Fatalf("expected adaptation to switch while protection is bypassed, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	assertAdaptationSampled(t, ctx, true)
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok || policy.Action != routerLearningActionBypass || policy.Reason != "decision_bypass" {
		t.Fatalf("expected protection bypass policy, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestRouterLearningProtectionMissingIdentityDoesNotSuppressAdaptationSampling(t *testing.T) {
	originalSeedSource := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = originalSeedSource })

	router := &OpenAIRouter{Config: routerLearningTestConfig(config.RouterLearningScopeConversation)}
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"frontier",
		routerLearningOutcomeGoodFit,
		1000,
	)
	router.routerLearningRuntimeState().recordModelExperience(
		"adaptive",
		2,
		"cheap",
		routerLearningOutcomeUnderpowered,
		1000,
	)
	ctx := &RequestContext{Headers: map[string]string{"x-session-id": "session-a"}}
	ctx.VSRSelectedDecision = &config.Decision{Name: "adaptive", Tier: 2}
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
		t.Fatalf("expected adaptation to switch when protection identity is missing, result=%#v selected=%#v applied=%v", result, selected, applied)
	}
	assertAdaptationSampled(t, ctx, true)
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok || policy.Action != routerLearningActionSuppressSampling || policy.Reason != "missing_identity" {
		t.Fatalf("expected missing identity protection diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func assertAdaptationSampled(t *testing.T, ctx *RequestContext, want bool) {
	t.Helper()
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation)
	if !ok {
		t.Fatalf("expected adaptation policy, got %#v", ctx.VSRLearningPolicies)
	}
	sampling, ok := policy.ToMap()["sampling"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected adaptation sampling diagnostics, got %#v", policy.ToMap())
	}
	if sampling["used"] != want {
		t.Fatalf("expected sampling used=%v, got %#v", want, sampling)
	}
}
