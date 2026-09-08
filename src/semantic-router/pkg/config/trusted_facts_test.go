package config

import "testing"

func TestTrustedFactsValidate(t *testing.T) {
	// Disabled/nil is a no-op (existing behavior unchanged).
	var nilCfg *TrustedFactsConfig
	if err := nilCfg.Validate(); err != nil {
		t.Fatalf("nil should pass: %v", err)
	}
	if err := (&TrustedFactsConfig{Enabled: false, TrustSources: []string{"bogus"}}).Validate(); err != nil {
		t.Fatalf("disabled should skip checks: %v", err)
	}
	// Valid authoritative config.
	valid := &TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      TrustedEnforcementAuthoritative,
		TrustSources:     []string{TrustedSourceOperatorPolicy, TrustedSourceGatewayAttested},
		FreshnessSeconds: 300,
		StageRoles:       []string{TrustedStageCandidate, TrustedStageFinal},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid should pass: %v", err)
	}
	// Advisory default (empty enforcement) is observe-only and valid.
	advisory := &TrustedFactsConfig{
		Enabled:      true,
		TrustSources: []string{TrustedSourceOperatorPolicy},
		StageRoles:   []string{TrustedStageCandidate},
	}
	if got := advisory.EffectiveEnforcement(); got != TrustedEnforcementAdvisory {
		t.Fatalf("empty enforcement should default to advisory, got %q", got)
	}
	if err := advisory.Validate(); err != nil {
		t.Fatalf("advisory default should pass: %v", err)
	}
	// Reject untrusted sources (prompt/model-output must never authorize).
	untrusted := &TrustedFactsConfig{
		Enabled:      true,
		TrustSources: []string{"prompt"},
		StageRoles:   []string{TrustedStageCandidate},
	}
	if err := untrusted.Validate(); err == nil {
		t.Fatalf("untrusted source should fail")
	}
	// Reject missing sources/roles and bad ranges.
	cases := []TrustedFactsConfig{
		{Enabled: true, TrustSources: []string{}, StageRoles: []string{TrustedStageCandidate}},
		{Enabled: true, TrustSources: []string{TrustedSourceOperatorPolicy}, StageRoles: []string{}},
		{Enabled: true, Enforcement: "bogus", TrustSources: []string{TrustedSourceOperatorPolicy}, StageRoles: []string{TrustedStageCandidate}},
		{Enabled: true, TrustSources: []string{TrustedSourceOperatorPolicy}, StageRoles: []string{TrustedStageCandidate}, FreshnessSeconds: -1},
		{Enabled: true, TrustSources: []string{TrustedSourceOperatorPolicy}, StageRoles: []string{TrustedStageCandidate}, FreshnessSeconds: 99999},
		{Enabled: true, TrustSources: []string{TrustedSourceOperatorPolicy}, StageRoles: []string{"unknown-stage"}},
	}
	for i, tc := range cases {
		if err := tc.Validate(); err == nil {
			t.Fatalf("case %d should fail: %+v", i, tc)
		}
	}
}

func TestToolsPluginTrustedFactsWiring(t *testing.T) {
	// TrustedFacts nil preserves existing behavior.
	base := &ToolsPluginConfig{Enabled: true, Mode: ToolsPluginModePassthrough}
	if base.TrustedFactsEnabled() {
		t.Fatalf("nil TrustedFacts should report disabled")
	}
	if err := base.Validate(); err != nil {
		t.Fatalf("base without trusted facts should pass: %v", err)
	}
	// Enabled trusted facts flow through ToolsPluginConfig.Validate.
	withTrusted := &ToolsPluginConfig{
		Enabled: true,
		Mode:    ToolsPluginModePassthrough,
		TrustedFacts: &TrustedFactsConfig{
			Enabled:      true,
			TrustSources: []string{TrustedSourceOperatorPolicy},
			StageRoles:   []string{TrustedStageFinal},
		},
	}
	if !withTrusted.TrustedFactsEnabled() {
		t.Fatalf("should report enabled")
	}
	if err := withTrusted.Validate(); err != nil {
		t.Fatalf("valid trusted facts should pass: %v", err)
	}
	withBad := &ToolsPluginConfig{
		Enabled: true,
		Mode:    ToolsPluginModePassthrough,
		TrustedFacts: &TrustedFactsConfig{
			Enabled:      true,
			TrustSources: []string{"client-metadata"},
			StageRoles:   []string{TrustedStageFinal},
		},
	}
	if err := withBad.Validate(); err == nil {
		t.Fatalf("untrusted source via ToolsPluginConfig should fail")
	}
}
