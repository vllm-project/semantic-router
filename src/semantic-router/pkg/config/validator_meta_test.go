package config

import (
	"strings"
	"testing"
)

func TestParseRoutingYAMLBytesAcceptsMetaRoutingConfig(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    domains:
      - name: general
        description: General requests
  meta:
    mode: shadow
    max_passes: 2
    trigger_policy:
      decision_margin_below: 0.18
      projection_boundary_within: 0.07
      partition_conflict: true
      required_families:
        - type: preference
          min_confidence: 0.65
          min_matches: 1
      family_disagreements:
        - cheap: keyword
          expensive: embedding
    allowed_actions:
      - type: disable_compression
      - type: rerun_signal_families
        signal_families: [preference, jailbreak]
  decisions:
    - name: general_route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3-8b
`)

	cfg, err := ParseRoutingYAMLBytes(yaml)
	if err != nil {
		t.Fatalf("ParseRoutingYAMLBytes returned error: %v", err)
	}

	if cfg.MetaRouting.Mode != MetaRoutingModeShadow {
		t.Fatalf("expected mode %q, got %q", MetaRoutingModeShadow, cfg.MetaRouting.Mode)
	}
	if cfg.MetaRouting.MaxPasses != 2 {
		t.Fatalf("expected max_passes 2, got %d", cfg.MetaRouting.MaxPasses)
	}
	if cfg.MetaRouting.TriggerPolicy == nil || len(cfg.MetaRouting.AllowedActions) != 2 {
		t.Fatalf("expected trigger policy and 2 actions, got %#v", cfg.MetaRouting)
	}
}

func TestParseRoutingYAMLBytesRejectsConfiguredMetaRoutingWithoutMode(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    domains:
      - name: general
        description: General requests
  meta:
    max_passes: 2
  decisions:
    - name: general_route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3-8b
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err == nil {
		t.Fatal("expected ParseRoutingYAMLBytes to reject configured routing.meta without mode")
	}
	if !strings.Contains(err.Error(), "routing.meta.mode is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseRoutingYAMLBytesRejectsUnknownMetaRoutingSignalFamily(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    domains:
      - name: general
        description: General requests
  meta:
    mode: observe
    trigger_policy:
      required_families:
        - type: unknown_family
  decisions:
    - name: general_route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3-8b
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err == nil {
		t.Fatal("expected ParseRoutingYAMLBytes to reject unknown required family")
	}
	if !strings.Contains(err.Error(), "unknown_family") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCanonicalRoutingFromRouterConfigPreservesMetaRouting(t *testing.T) {
	decisionMargin := 0.12
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			MetaRouting: MetaRoutingConfig{
				Mode:      MetaRoutingModeActive,
				MaxPasses: 2,
				TriggerPolicy: &MetaTriggerPolicy{
					DecisionMarginBelow: &decisionMargin,
				},
				AllowedActions: []MetaRefinementAction{
					{Type: MetaRoutingActionDisableCompression},
				},
			},
		},
	}

	canonical := CanonicalRoutingFromRouterConfig(cfg)
	if canonical.Meta.Mode != MetaRoutingModeActive {
		t.Fatalf("expected canonical meta mode %q, got %q", MetaRoutingModeActive, canonical.Meta.Mode)
	}
	if len(canonical.Meta.AllowedActions) != 1 {
		t.Fatalf("expected one canonical meta action, got %d", len(canonical.Meta.AllowedActions))
	}
	if canonical.Meta.TriggerPolicy == nil || canonical.Meta.TriggerPolicy.DecisionMarginBelow == nil {
		t.Fatalf("expected canonical trigger policy to be preserved, got %#v", canonical.Meta.TriggerPolicy)
	}
}
