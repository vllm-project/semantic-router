package config

import "testing"

func TestUnknownPolicyValues(t *testing.T) {
	for _, policy := range UnknownPolicies {
		if !policy.IsValid() {
			t.Fatalf("%q must be valid", policy)
		}
	}
	if UnknownPolicy("allow").IsValid() || UnknownPolicy("").IsValid() {
		t.Fatal("unlisted values must be invalid")
	}
	if got := UnknownPolicyChoices(); got != "no_match, match, or fail_request" {
		t.Fatalf("choices = %q", got)
	}
}
