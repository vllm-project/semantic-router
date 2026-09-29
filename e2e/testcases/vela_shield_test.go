package testcases

import (
	"net/http"
	"testing"
)

func TestVelaShieldVerdictRequiresMatchedRuleAndDecisionToAgree(t *testing.T) {
	headers := func(matched, decision string) http.Header {
		h := http.Header{}
		if matched != "" {
			h.Set("x-vsr-matched-safety", matched)
		}
		h.Set("x-vsr-selected-decision", decision)
		return h
	}
	unsafe, benign := velaShieldProbes[0], velaShieldProbes[1]
	for _, tc := range []struct {
		name  string
		probe velaShieldProbe
		h     http.Header
		ok    bool
	}{
		{"unsafe matched", unsafe, headers("other, unsafe-content", velaShieldSafetyDecision), true},
		{"unsafe missed", unsafe, headers("", velaShieldDefaultRoute), false},
		{"unsafe matched without decision", unsafe, headers("unsafe-content", velaShieldDefaultRoute), false},
		{"benign passed", benign, headers("", velaShieldDefaultRoute), true},
		{"benign flagged", benign, headers("unsafe-content", velaShieldSafetyDecision), false},
		{"benign on unexpected route", benign, headers("", ""), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := checkVelaShieldVerdict(tc.probe, tc.h); (err == nil) != tc.ok {
				t.Fatalf("verdict error = %v, want ok=%v", err, tc.ok)
			}
		})
	}
}
