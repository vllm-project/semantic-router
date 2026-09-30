package config

import (
	"strings"
	"testing"
)

func fusionFallbackTestConfig(fusion *FusionAlgorithmConfig) (*RouterConfig, Decision) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			VLLMEndpoints: []VLLMEndpoint{
				{Name: "backup-endpoint", Model: "backup-model", Address: "127.0.0.1", Port: 8000},
				{Name: "panel-endpoint", Model: "panel-a", Address: "127.0.0.1", Port: 8001},
				{Name: "anthropic-endpoint", Model: "claude-model", Address: "127.0.0.1", Port: 8002},
				{Name: "image-endpoint", Model: "image-model", Address: "127.0.0.1", Port: 8003},
			},
			ModelConfig: map[string]ModelParams{
				"backup-model": {},
				"panel-a":      {},
				"claude-model": {APIFormat: ClientProtocolAnthropic},
				"image-model":  {Modality: "image"},
			},
		},
	}
	decision := Decision{
		Name:      "fusion_route",
		Algorithm: &AlgorithmConfig{Type: DecisionAlgorithmFusion, Fusion: fusion},
	}
	return cfg, decision
}

func TestFusionFallbackTargetAccepted(t *testing.T) {
	cases := []struct {
		name   string
		fusion *FusionAlgorithmConfig
	}{
		{name: "no fusion block"},
		{
			name:   "default policy skips target validation",
			fusion: &FusionAlgorithmConfig{AnalysisModels: []string{"panel-a"}},
		},
		{
			name: "explicit fail skips target validation",
			fusion: &FusionAlgorithmConfig{
				QuorumFailurePolicy: FusionQuorumFailurePolicyFail,
				AnalysisModels:      []string{"panel-a"},
			},
		},
		{
			name: "concrete provider model",
			fusion: &FusionAlgorithmConfig{
				QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: "backup-model",
				AnalysisModels:       []string{"panel-a"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, decision := fusionFallbackTestConfig(tc.fusion)
			if err := validateDecisionFusionFallbackTarget(cfg, decision); err != nil {
				t.Fatalf("validateDecisionFusionFallbackTarget() = %v, want nil", err)
			}
		})
	}
}

func TestFusionFallbackTargetRejected(t *testing.T) {
	cases := []struct {
		name       string
		target     string
		wantReason string
	}{
		{name: "undeclared model", target: "missing-model", wantReason: "routing.modelCards"},
		{name: "anthropic api format", target: "claude-model", wantReason: "OpenAI-compatible"},
		{name: "non chat modality", target: "image-model", wantReason: "chat-capable"},
		{name: "panel member", target: "panel-a", wantReason: "analysis models"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
				QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: tc.target,
				AnalysisModels:       []string{"panel-a"},
			})
			err := validateDecisionFusionFallbackTarget(cfg, decision)
			if err == nil {
				t.Fatal("validateDecisionFusionFallbackTarget() = nil, want error")
			}
			if !strings.Contains(err.Error(), tc.wantReason) {
				t.Fatalf("error %q must explain %q", err.Error(), tc.wantReason)
			}
		})
	}
}

// A fallback target that re-enters a composite path would run the same panel
// again, so every virtual and composite slug must be rejected.
func TestFusionFallbackTargetRejectsCompositeSlugs(t *testing.T) {
	cases := []struct {
		name  string
		apply func(cfg *RouterConfig, slug string)
	}{
		{
			name: "fusion slug",
			apply: func(cfg *RouterConfig, slug string) {
				cfg.Looper.Fusion.ModelNames = []string{slug}
			},
		},
		{
			name: "flow slug",
			apply: func(cfg *RouterConfig, slug string) {
				cfg.Looper.Flow.ModelNames = []string{slug}
			},
		},
		{
			name: "remom slug",
			apply: func(cfg *RouterConfig, slug string) {
				cfg.Looper.ReMoM.ModelNames = []string{slug}
			},
		},
	}
	const slug = "backup-model"
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
				QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget: slug,
				AnalysisModels:       []string{"panel-a"},
			})
			tc.apply(cfg, slug)
			err := validateDecisionFusionFallbackTarget(cfg, decision)
			if err == nil {
				t.Fatal("composite fallback target must be rejected")
			}
			if !strings.Contains(err.Error(), "composite or virtual") {
				t.Fatalf("error %q must explain the composite guard", err.Error())
			}
		})
	}
}

// A routing-only fragment carries no provider state, so provider-dependent
// checks must not make it undecompilable. This is the DSL round-trip path.
func TestFusionFallbackTargetAllowsRoutingOnlyFragment(t *testing.T) {
	yaml := []byte(`
routing:
  modelCards:
    - name: panel-a
    - name: panel-b
    - name: backup-model
  decisions:
    - name: fusion_quorum
      priority: 10
      modelRefs:
        - model: panel-a
          use_reasoning: false
        - model: panel-b
          use_reasoning: false
      algorithm:
        type: fusion
        fusion:
          model: judge
          analysis_models: [panel-a, panel-b]
          min_successful_responses: 2
          quorum_failure_policy: fallback
          quorum_fallback_target: backup-model
`)
	if _, err := ParseRoutingYAMLBytes(yaml); err != nil {
		t.Fatalf("routing-only fragment must round-trip, got %v", err)
	}
}

// Endpoint count is not a routing-fragment discriminator: a complete config
// with model cards but no providers must still reject a backendless target.
func TestFusionFallbackTargetRejectsBackendlessTargetWithoutEndpoints(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"panel-a"},
	})
	cfg.VLLMEndpoints = nil

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("a complete config without providers must still reject a backendless target")
	}
	if !strings.Contains(err.Error(), "provider backend") {
		t.Fatalf("error %q must explain the missing backend", err.Error())
	}
}

// The effective panel is modelRefs when analysis_models is omitted, so
// capability coverage must be computed from it too.
func TestFusionFallbackTargetChecksCapabilitiesAgainstModelRefPanel(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
	})
	decision.ModelRefs = []ModelRef{{Model: "panel-a"}}
	cfg.ModelConfig["panel-a"] = ModelParams{Capabilities: []string{"tools", "vision"}}
	cfg.ModelConfig["backup-model"] = ModelParams{Capabilities: []string{"tools"}}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("capabilities from a modelRefs panel must be enforced")
	}
	if !strings.Contains(err.Error(), "vision") {
		t.Fatalf("error %q must name the missing capability", err.Error())
	}
}

// A complete config still gets the provider-backend check.
func TestFusionFallbackTargetRequiresBackendInCompleteConfig(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backendless-model",
		AnalysisModels:       []string{"panel-a"},
	})
	cfg.ModelConfig["backendless-model"] = ModelParams{}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("a declared model without a provider backend must be rejected")
	}
	if !strings.Contains(err.Error(), "provider backend") {
		t.Fatalf("error %q must explain the missing backend", err.Error())
	}
}

// The fallback answers in place of the whole panel, so it must declare at least
// the capabilities the panel declares.
func TestFusionFallbackTargetRejectsMissingCapabilities(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"panel-a"},
	})
	cfg.ModelConfig["panel-a"] = ModelParams{Capabilities: []string{"tools", "vision"}}
	cfg.ModelConfig["backup-model"] = ModelParams{Capabilities: []string{"tools"}}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("a fallback target missing a panel capability must be rejected")
	}
	if !strings.Contains(err.Error(), "vision") {
		t.Fatalf("error %q must name the missing capability", err.Error())
	}
}

// A target that declares no capabilities cannot be shown compatible with a panel
// that declares some, so it is rejected. Accepting it would let any target pass
// validation simply by omitting its declaration.
func TestFusionFallbackTargetRejectedWhenCapabilitiesUndeclared(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"panel-a"},
	})
	cfg.ModelConfig["panel-a"] = ModelParams{Capabilities: []string{"tools"}}

	err := validateDecisionFusionFallbackTarget(cfg, decision)
	if err == nil {
		t.Fatal("an undeclared target must not satisfy a panel that declares capabilities")
	}
	if !strings.Contains(err.Error(), "no declared capabilities") {
		t.Fatalf("error should explain the undeclared target, got %v", err)
	}
}

// When the panel declares nothing there is no requirement to meet, so a target
// declaring nothing is accepted. The rule constrains what the panel needs, not
// metadata completeness in general.
func TestFusionFallbackTargetAcceptedWhenPanelDeclaresNothing(t *testing.T) {
	cfg, decision := fusionFallbackTestConfig(&FusionAlgorithmConfig{
		QuorumFailurePolicy:  FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
		AnalysisModels:       []string{"panel-a"},
	})
	cfg.ModelConfig["panel-a"] = ModelParams{}

	if err := validateDecisionFusionFallbackTarget(cfg, decision); err != nil {
		t.Fatalf("a panel declaring no capabilities imposes no requirement, got %v", err)
	}
}

// An empty model catalog in a complete config means the target is undeclared,
// not that catalog rules should be skipped. Driven through the public parser so
// the whole load path is covered, not just the helper.
func TestFusionFallbackTargetRejectedByPublicParserWithEmptyCatalog(t *testing.T) {
	yaml := []byte(`
version: v0.3
routing:
  decisions:
    - name: fusion_quorum
      priority: 10
      modelRefs:
        - model: panel-a
          use_reasoning: false
        - model: panel-b
          use_reasoning: false
      algorithm:
        type: fusion
        fusion:
          model: judge
          analysis_models: [panel-a, panel-b]
          min_successful_responses: 2
          quorum_failure_policy: fallback
          quorum_fallback_target: backup-model
`)
	if _, err := ParseYAMLBytes(yaml); err == nil {
		t.Fatal("complete config accepted a fallback target with no model card or provider")
	}
}
