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

package lookuptable_test

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func TestKeyString_QualityGap(t *testing.T) {
	k := lookuptable.QualityGapKey("coding", "gpt-4", "claude-3")
	want := "quality_gap::coding::gpt-4::claude-3"
	if got := k.String(); got != want {
		t.Errorf("QualityGapKey.String() = %q, want %q", got, want)
	}
}

func TestKeyString_HandoffPenalty(t *testing.T) {
	k := lookuptable.HandoffPenaltyKey("gpt-4", "claude-3")
	want := "handoff_penalty::gpt-4::claude-3"
	if got := k.String(); got != want {
		t.Errorf("HandoffPenaltyKey.String() = %q, want %q", got, want)
	}
}

func TestKeyString_RemainingTurnPrior(t *testing.T) {
	k := lookuptable.RemainingTurnPriorKey("customer_support")
	want := "remaining_turn_prior::customer_support"
	if got := k.String(); got != want {
		t.Errorf("RemainingTurnPriorKey.String() = %q, want %q", got, want)
	}
}

func TestParseKey_RoundTrip(t *testing.T) {
	keys := []lookuptable.Key{
		lookuptable.QualityGapKey("coding", "gpt-4", "claude-3"),
		lookuptable.HandoffPenaltyKey("gpt-4", "claude-3"),
		lookuptable.RemainingTurnPriorKey("customer_support"),
	}

	for _, k := range keys {
		s := k.String()
		parsed, err := lookuptable.ParseKey(s)
		if err != nil {
			t.Errorf("ParseKey(%q) unexpected error: %v", s, err)
			continue
		}
		if parsed.String() != s {
			t.Errorf("ParseKey(%q).String() = %q, want %q", s, parsed.String(), s)
		}
	}
}

func TestParseKey_Errors(t *testing.T) {
	bad := []string{
		"",
		"quality_gap",
		"quality_gap::only_one_extra",
		"handoff_penalty::only_one",
		"remaining_turn_prior",
		"unknown_table::x",
	}

	for _, s := range bad {
		if _, err := lookuptable.ParseKey(s); err == nil {
			t.Errorf("ParseKey(%q) expected error, got nil", s)
		}
	}
}
