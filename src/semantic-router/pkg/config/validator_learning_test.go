package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesAcceptsDecisionSessionAwareOverrides(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
      adaptations:
        session_aware:
          mode: observe
          scope: session
          tuning:
            switch_margin: 0.11
            cache_weight: 0.25
global:
  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
          scope: conversation
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	adaptation := cfg.Decisions[0].Adaptations.SessionAware
	if adaptation == nil {
		t.Fatal("expected decision session_aware adaptation override")
	}
	if adaptation.Mode != DecisionAdaptationModeObserve || adaptation.Scope != RouterLearningScopeSession {
		t.Fatalf("unexpected decision adaptation override: %#v", adaptation)
	}
	if adaptation.Tuning.SwitchMargin == nil || *adaptation.Tuning.SwitchMargin != 0.11 {
		t.Fatalf("expected decision switch_margin override, got %#v", adaptation.Tuning.SwitchMargin)
	}
	if adaptation.Tuning.CacheWeight == nil || *adaptation.Tuning.CacheWeight != 0.25 {
		t.Fatalf("expected decision cache_weight override, got %#v", adaptation.Tuning.CacheWeight)
	}
}

func TestParseYAMLBytesAcceptsLearningAdaptationControls(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
      adaptations:
        bandit:
          mode: observe
          scope: decision
          goals:
            quality: 1.0
            cost: 0.25
          tuning:
            exploration_budget: 0.02
        elo:
          mode: apply
        personalization:
          mode: bypass
global:
  router:
    learning:
      enabled: true
      adaptations:
        bandit:
          enabled: true
          algorithm: linucb
          scope: decision
          goals:
            quality: 1.0
            cost: 0.25
            latency: 0.1
          tuning:
            exploration_budget: 0.05
        elo:
          enabled: true
          scope: decision
          initial_rating: 1200
          k_factor: 32
        personalization:
          enabled: true
          scope: session
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.RouterLearning.Adaptations.Bandit.EffectiveAlgorithm() != RouterLearningBanditAlgorithmLinUCB {
		t.Fatalf("expected bandit algorithm linucb, got %q", cfg.RouterLearning.Adaptations.Bandit.EffectiveAlgorithm())
	}
	if cfg.RouterLearning.Adaptations.Bandit.EffectiveScope() != RouterLearningScopeDecision {
		t.Fatalf("expected bandit scope decision, got %q", cfg.RouterLearning.Adaptations.Bandit.EffectiveScope())
	}
	if cfg.Decisions[0].Adaptations.Bandit == nil || cfg.Decisions[0].Adaptations.Bandit.Mode != DecisionAdaptationModeObserve {
		t.Fatalf("expected decision bandit observe override, got %#v", cfg.Decisions[0].Adaptations.Bandit)
	}
	if cfg.Decisions[0].Adaptations.Elo == nil || cfg.Decisions[0].Adaptations.Elo.Mode != DecisionAdaptationModeApply {
		t.Fatalf("expected decision elo apply override, got %#v", cfg.Decisions[0].Adaptations.Elo)
	}
	if cfg.Decisions[0].Adaptations.Personalization == nil || cfg.Decisions[0].Adaptations.Personalization.Mode != DecisionAdaptationModeBypass {
		t.Fatalf("expected decision personalization bypass override, got %#v", cfg.Decisions[0].Adaptations.Personalization)
	}
}

func TestParseYAMLBytesRejectsUnknownDecisionAdaptation(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
      adaptations:
        unknown_learning:
          mode: apply
global:
  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
`))
	if err == nil {
		t.Fatal("expected unknown decision adaptation to be rejected")
	}
	if !strings.Contains(err.Error(), "routing.decisions[0].adaptations.unknown_learning") {
		t.Fatalf("expected unknown adaptation path in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsMigratedLearningAlgorithms(t *testing.T) {
	for name, algorithmType := range map[string]string{
		"elo":       "elo",
		"rl_driven": "rl_driven",
		"gmtrouter": "gmtrouter",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
      algorithm:
        type: ` + algorithmType + `
`))
			if err == nil {
				t.Fatalf("expected algorithm.type=%s to be rejected", algorithmType)
			}
			if !strings.Contains(err.Error(), "global.router.learning.adaptations") {
				t.Fatalf("expected Router Learning migration error, got %v", err)
			}
		})
	}
}

func TestParseYAMLBytesRejectsUnknownSessionAwareFields(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
global:
  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
          privacy_affinity: true
`))
	if err == nil {
		t.Fatal("expected unknown global session_aware field to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.adaptations.session_aware.privacy_affinity") {
		t.Fatalf("expected unknown global session-aware path in error, got %v", err)
	}
}
