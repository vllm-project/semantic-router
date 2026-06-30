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
	"math/rand"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// linucbContextualConfig builds a router config with linucb enabled and a
// disabled protection layer so the bandit can score freely.
func linucbContextualConfig(strategy string) *config.RouterConfig {
	disabled := false
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "fast",
			ModelConfig: map[string]config.ModelParams{
				"fast":    {},
				"smart":   {},
				"premium": {},
			},
		},
		RouterLearning: config.RouterLearningConfig{
			Enabled: true,
			Adaptation: config.RouterLearningAdaptationConfig{
				CandidateSet: config.RouterLearningCandidateSetDecision,
				Strategy:     strategy,
			},
			Protection: config.RouterLearningProtectionConfig{
				Enabled: &disabled,
			},
		},
	}
	dim := 16
	alpha := 1.0
	lambda := 1.0
	switch strategy {
	case config.RouterLearningStrategyLinUCB:
		cfg.RouterLearning.Adaptation.LinUCB = &config.RouterLearningLinUCBConfig{
			Dim:    &dim,
			Alpha:  &alpha,
			Lambda: &lambda,
		}
	case config.RouterLearningStrategyLinearThompson:
		sigma := 0.3
		cfg.RouterLearning.Adaptation.LinearThompson = &config.RouterLearningLinearThompsonConfig{
			Dim:    &dim,
			Sigma:  &sigma,
			Lambda: &lambda,
		}
	}
	return cfg
}

// TestLinUCBStrategyIsRegistered guards against a regression where the
// strategy registry quietly drops linucb (causing the runtime to fall
// back to "strategy_unavailable").
func TestLinUCBStrategyIsRegistered(t *testing.T) {
	cfg := config.RouterLearningAdaptationConfig{Strategy: config.RouterLearningStrategyLinUCB}
	strategy, ok := routerLearningAdaptationStrategies.Strategy(cfg)
	if !ok {
		t.Fatalf("linucb strategy not registered")
	}
	if got := strategy.Name(); got != config.RouterLearningStrategyLinUCB {
		t.Fatalf("strategy name = %q, want %q", got, config.RouterLearningStrategyLinUCB)
	}
}

// TestLinearThompsonStrategyIsRegistered mirrors the LinUCB registration check.
func TestLinearThompsonStrategyIsRegistered(t *testing.T) {
	cfg := config.RouterLearningAdaptationConfig{Strategy: config.RouterLearningStrategyLinearThompson}
	strategy, ok := routerLearningAdaptationStrategies.Strategy(cfg)
	if !ok {
		t.Fatalf("linear_thompson strategy not registered")
	}
	if got := strategy.Name(); got != config.RouterLearningStrategyLinearThompson {
		t.Fatalf("strategy name = %q, want %q", got, config.RouterLearningStrategyLinearThompson)
	}
}

// TestLinUCBLearnsContextDependentArmFromOutcomes is the paper-benchmark-style
// convergence test. We construct a 2-cluster contextual workload where:
//   - "code" queries should be answered by `smart`
//   - "story" queries should be answered by `fast`
// We feed a few hundred synthetic outcomes (good_fit on the matching arm,
// underpowered on the other arms) and assert that LinUCB's posterior mean
// favours the matching arm at score time, for each cluster.
func TestLinUCBLearnsContextDependentArmFromOutcomes(t *testing.T) {
	router := &OpenAIRouter{Config: linucbContextualConfig(config.RouterLearningStrategyLinUCB)}
	rt := router.routerLearningRuntimeState()

	// Drive 400 synthetic outcomes through the contextual state directly.
	// We don't go through applyLinUCBAdaptation because the protection /
	// recorder pipeline isn't relevant to the convergence claim.
	state := rt.contextualState(config.RouterLearningStrategyLinUCB, 16, 1.0)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 400; i++ {
		var query string
		var armWanted string
		var armOther string
		if i%2 == 0 {
			query = "Please write a Python function that sorts a list."
			armWanted = "smart"
			armOther = "fast"
		} else {
			query = "Tell me a short whimsical story about a cat."
			armWanted = "fast"
			armOther = "smart"
		}
		x := extractContextFeatures(&selection.SelectionContext{Query: query}, 16)
		// Reward = 1.0 for the correct arm; small noise on the wrong arm.
		correctArm := state.arm(contextualBanditKey(config.RouterLearningStrategyLinUCB, "", 0, armWanted))
		wrongArm := state.arm(contextualBanditKey(config.RouterLearningStrategyLinUCB, "", 0, armOther))
		_ = correctArm.update(x, 1.0)
		_ = wrongArm.update(x, 0.0+0.05*rng.NormFloat64())
	}

	// Now score on a fresh code query — `smart` should out-score `fast`.
	codeCtx := &selection.SelectionContext{
		Query: "Write a Go function to compute Fibonacci numbers iteratively.",
		CandidateModels: []config.ModelRef{
			{Model: "fast"},
			{Model: "smart"},
		},
	}
	codeX := extractContextFeatures(codeCtx, 16)
	fastArm := state.arm(contextualBanditKey(config.RouterLearningStrategyLinUCB, "", 0, "fast"))
	smartArm := state.arm(contextualBanditKey(config.RouterLearningStrategyLinUCB, "", 0, "smart"))
	if smartArm.dotTheta(codeX) <= fastArm.dotTheta(codeX) {
		t.Fatalf("expected LinUCB to prefer smart on a code query, got smart=%v fast=%v",
			smartArm.dotTheta(codeX), fastArm.dotTheta(codeX))
	}

	// And on a story query — `fast` should win.
	storyCtx := &selection.SelectionContext{
		Query: "Tell me a fairy tale about a friendly dragon and a baker.",
		CandidateModels: []config.ModelRef{
			{Model: "fast"},
			{Model: "smart"},
		},
	}
	storyX := extractContextFeatures(storyCtx, 16)
	if fastArm.dotTheta(storyX) <= smartArm.dotTheta(storyX) {
		t.Fatalf("expected LinUCB to prefer fast on a story query, got fast=%v smart=%v",
			fastArm.dotTheta(storyX), smartArm.dotTheta(storyX))
	}
}

// TestLinUCBPolicyEmitsContextualHyperparams asserts that the replay policy
// map includes dim/alpha/lambda diagnostics so a downstream reader can tell
// the strategy apart from routing_sampling.
func TestLinUCBPolicyEmitsContextualHyperparams(t *testing.T) {
	diag := &routerLearningAdaptationDiagnostics{
		strategy: config.RouterLearningStrategyLinUCB,
		dim:      32,
		alpha:    0.7,
		lambda:   2.0,
	}
	out := diag.toPolicyMap()
	if dim, _ := out["dim"].(int); dim != 32 {
		t.Fatalf("expected dim=32 in policy map, got %v", out["dim"])
	}
	if alpha, _ := out["alpha"].(float64); alpha != 0.7 {
		t.Fatalf("expected alpha=0.7 in policy map, got %v", out["alpha"])
	}
	if lambda, _ := out["lambda"].(float64); lambda != 2.0 {
		t.Fatalf("expected lambda=2.0 in policy map, got %v", out["lambda"])
	}
	if _, ok := out["sigma"]; ok {
		t.Fatalf("LinUCB diagnostics should not emit sigma; got %v", out)
	}
}

// TestLinearThompsonPolicyEmitsSigma checks the Linear Thompson Sampling
// strategy reports sigma instead of alpha.
func TestLinearThompsonPolicyEmitsSigma(t *testing.T) {
	diag := &routerLearningAdaptationDiagnostics{
		strategy: config.RouterLearningStrategyLinearThompson,
		dim:      32,
		sigma:    0.4,
		lambda:   1.0,
	}
	out := diag.toPolicyMap()
	if sigma, _ := out["sigma"].(float64); sigma != 0.4 {
		t.Fatalf("expected sigma=0.4 in policy map, got %v", out["sigma"])
	}
	if _, ok := out["alpha"]; ok {
		t.Fatalf("Linear Thompson diagnostics should not emit alpha; got %v", out)
	}
}

// TestExtractContextFeaturesIsDeterministic asserts that two equal queries
// produce equal feature vectors. This is the baseline contract the bandit
// state relies on for replay correctness.
func TestExtractContextFeaturesIsDeterministic(t *testing.T) {
	queries := []string{
		"Tell me a story",
		"Write Python code",
		"",
		strings.Repeat("hello world ", 50),
	}
	for _, q := range queries {
		a := extractContextFeatures(&selection.SelectionContext{Query: q}, 32)
		b := extractContextFeatures(&selection.SelectionContext{Query: q}, 32)
		if len(a) != len(b) {
			t.Fatalf("dimension mismatch: %d vs %d", len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("non-deterministic feature at idx %d for query %q: %v vs %v", i, q, a[i], b[i])
			}
		}
	}
}

// TestExtractContextFeaturesDifferentiatesCodeFromStory asserts that the
// feature vectors for "code" and "story" prompts are not identical — the
// bandit cannot learn cluster-conditional rewards if the feature extractor
// erases the cluster signal.
func TestExtractContextFeaturesDifferentiatesCodeFromStory(t *testing.T) {
	code := extractContextFeatures(&selection.SelectionContext{
		Query: "Write a Python function with a list comprehension and explain the time complexity.",
	}, 32)
	story := extractContextFeatures(&selection.SelectionContext{
		Query: "Tell me a fairy tale about a baker dragon village.",
	}, 32)
	identical := true
	for i := range code {
		if code[i] != story[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatalf("feature extractor produced identical vectors for code vs story queries")
	}
}

// TestContextualStateScopedByStrategy guards a tricky aspect of state
// management: LinUCB and Linear Thompson must NOT share matrix state, since
// changing the strategy mid-deployment would otherwise ingest poisoned
// posterior estimates from a different update rule.
func TestContextualStateScopedByStrategy(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	linucb := rt.contextualState(config.RouterLearningStrategyLinUCB, 8, 1.0)
	linthompson := rt.contextualState(config.RouterLearningStrategyLinearThompson, 8, 1.0)
	if linucb == linthompson {
		t.Fatalf("expected per-strategy contextual state isolation; got shared state")
	}
}

// silence unused import errors when this file is the only consumer in CI subsets.
var _ = routerruntime.RouterOutcomeTargetModel
