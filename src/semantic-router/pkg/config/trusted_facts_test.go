package config

import (
	"strings"
	"testing"
)

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
		{Enabled: true, TrustSources: []string{TrustedSourceRuntimeFresh}, StageRoles: []string{TrustedStageCandidate}, FreshnessSeconds: 0},
	}
	for i, tc := range cases {
		if err := tc.Validate(); err == nil {
			t.Fatalf("case %d should fail: %+v", i, tc)
		}
	}
	// runtime-fresh with an explicit bound passes.
	freshBounded := &TrustedFactsConfig{
		Enabled:          true,
		TrustSources:     []string{TrustedSourceRuntimeFresh},
		StageRoles:       []string{TrustedStageCandidate},
		FreshnessSeconds: 60,
	}
	if err := freshBounded.Validate(); err != nil {
		t.Fatalf("runtime-fresh with bound should pass: %v", err)
	}
}

func TestToolsPluginTrustedFactsWiring(t *testing.T) {
	trusted := &TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  TrustedEnforcementAuthoritative,
		TrustSources: []string{TrustedSourceOperatorPolicy},
		StageRoles:   []string{TrustedStageCandidate},
	}
	if (&ToolsPluginConfig{Enabled: true, Policy: &ToolsPolicyConfig{}}).TrustedFactsEnabled() {
		t.Fatalf("a policy without trusted facts reports them disabled")
	}
	if !(&ToolsPluginConfig{Enabled: true, Policy: &ToolsPolicyConfig{TrustedFacts: trusted}}).TrustedFactsEnabled() {
		t.Fatalf("an enabled plugin's trusted facts are enabled")
	}
	// A disabled plugin applies no policy, so its trusted facts never engage
	// the gate.
	if (&ToolsPluginConfig{Policy: &ToolsPolicyConfig{TrustedFacts: trusted}}).TrustedFactsEnabled() {
		t.Fatalf("a disabled plugin must report trusted facts disabled")
	}
	bad := *trusted
	bad.TrustSources = []string{"client-metadata"}
	if err := (&ToolsPolicyConfig{TrustedFacts: &bad}).validate("policy"); err == nil || !strings.HasPrefix(err.Error(), "policy.trusted_facts: ") {
		t.Fatalf("an untrusted source fails the policy: %v", err)
	}
}
