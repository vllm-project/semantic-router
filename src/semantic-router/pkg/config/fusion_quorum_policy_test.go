package config

import "testing"

func TestFusionQuorumFailurePolicyValues(t *testing.T) {
	for _, policy := range FusionQuorumFailurePolicies {
		if !policy.IsValid() {
			t.Fatalf("%q must be valid", policy)
		}
	}
	if FusionQuorumFailurePolicy("degrade").IsValid() || FusionQuorumFailurePolicy("").IsValid() {
		t.Fatal("unlisted values must be invalid")
	}
	if got := FusionQuorumFailurePolicyChoices(); got != "fail, fallback, or best_available" {
		t.Fatalf("choices = %q", got)
	}
}

func TestValidateFusionQuorumFailurePolicyAccepted(t *testing.T) {
	cases := []struct {
		name   string
		policy FusionQuorumFailurePolicy
		target string
	}{
		{name: "unset keeps the conservative default"},
		{name: "explicit fail", policy: FusionQuorumFailurePolicyFail},
		{name: "fallback with target", policy: FusionQuorumFailurePolicyFallback, target: "backup-model"},
		{name: "best available opt-in", policy: FusionQuorumFailurePolicyBestAvailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &FusionAlgorithmConfig{QuorumFailurePolicy: tc.policy, QuorumFallbackTarget: tc.target}
			if err := ValidateFusionAlgorithmConfig(cfg); err != nil {
				t.Fatalf("ValidateFusionAlgorithmConfig() = %v, want nil", err)
			}
			req := &FusionRequestConfig{QuorumFailurePolicy: tc.policy, QuorumFallbackTarget: tc.target}
			if err := req.Validate(); err != nil {
				t.Fatalf("FusionRequestConfig.Validate() = %v, want nil", err)
			}
		})
	}
}

func TestValidateFusionQuorumFailurePolicyRejected(t *testing.T) {
	cases := []struct {
		name   string
		policy FusionQuorumFailurePolicy
		target string
	}{
		{name: "unsupported policy", policy: "degrade"},
		{name: "fallback without target", policy: FusionQuorumFailurePolicyFallback},
		{name: "fallback with blank target", policy: FusionQuorumFailurePolicyFallback, target: "   "},
		{name: "target without fallback policy", policy: FusionQuorumFailurePolicyFail, target: "backup-model"},
		{name: "target with unset policy", target: "backup-model"},
		{name: "target with best available", policy: FusionQuorumFailurePolicyBestAvailable, target: "backup-model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &FusionAlgorithmConfig{QuorumFailurePolicy: tc.policy, QuorumFallbackTarget: tc.target}
			if err := ValidateFusionAlgorithmConfig(cfg); err == nil {
				t.Fatal("ValidateFusionAlgorithmConfig() = nil, want error")
			}
			req := &FusionRequestConfig{QuorumFailurePolicy: tc.policy, QuorumFallbackTarget: tc.target}
			if err := req.Validate(); err == nil {
				t.Fatal("FusionRequestConfig.Validate() = nil, want error")
			}
		})
	}
}

// Quorum-failure policy is panel-level and must stay independent of the
// per-attempt on_error contract.
func TestFusionQuorumFailurePolicyIndependentOfOnError(t *testing.T) {
	cfg := &FusionAlgorithmConfig{
		OnError:                FusionOnErrorFail,
		QuorumFailurePolicy:    FusionQuorumFailurePolicyBestAvailable,
		MinSuccessfulResponses: 2,
	}
	if err := ValidateFusionAlgorithmConfig(cfg); err != nil {
		t.Fatalf("on_error and quorum_failure_policy must combine freely, got %v", err)
	}
}
