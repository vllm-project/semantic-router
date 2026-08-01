package services

import (
	"math"
	"testing"
)

// securityWant is the expected shape of a buildSecurityResponse result.
type securityWant struct {
	isJailbreak   bool
	riskScore     float64
	confidence    float64
	recommend     string
	detectionType string // "" means no detection entries expected
	reasoning     bool
}

func assertSecurityResponse(t *testing.T, resp *SecurityResponse, w securityWant) {
	t.Helper()
	if resp.IsJailbreak != w.isJailbreak {
		t.Errorf("IsJailbreak = %v, want %v", resp.IsJailbreak, w.isJailbreak)
	}
	if math.Abs(resp.RiskScore-w.riskScore) > 1e-6 {
		t.Errorf("RiskScore = %v, want %v", resp.RiskScore, w.riskScore)
	}
	if math.Abs(resp.Confidence-w.confidence) > 1e-6 {
		t.Errorf("Confidence = %v, want %v", resp.Confidence, w.confidence)
	}
	if resp.Recommendation != w.recommend {
		t.Errorf("Recommendation = %q, want %q", resp.Recommendation, w.recommend)
	}
	assertDetections(t, resp, w.detectionType)
	if (resp.Reasoning != "") != w.reasoning {
		t.Errorf("Reasoning present = %v, want %v", resp.Reasoning != "", w.reasoning)
	}
}

func assertDetections(t *testing.T, resp *SecurityResponse, detectionType string) {
	t.Helper()
	want := []string{}
	if detectionType != "" {
		want = []string{detectionType}
	}
	if !equalStrings(resp.DetectionTypes, want) {
		t.Errorf("DetectionTypes = %v, want %v", resp.DetectionTypes, want)
	}
	if !equalStrings(resp.PatternsDetected, want) {
		t.Errorf("PatternsDetected = %v, want %v", resp.PatternsDetected, want)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestBuildSecurityResponse(t *testing.T) {
	tests := []struct {
		name             string
		isJailbreak      bool
		jailbreakType    string
		confidence       float32
		riskScore        float32
		includeReasoning bool
		want             securityWant
	}{
		{
			// Issue #2591: a confident benign prediction must report risk_score =
			// P(jailbreak) = 0.0043, not the benign confidence 0.9957.
			name:        "benign prediction reports jailbreak probability and allows",
			isJailbreak: false, jailbreakType: "benign", confidence: 0.9957, riskScore: 0.0043,
			want: securityWant{riskScore: 0.0043, confidence: 0.9957, recommend: "allow"},
		},
		{
			name:        "jailbreak prediction reports high risk, blocks, and lists detection",
			isJailbreak: true, jailbreakType: "jailbreak", confidence: 0.95, riskScore: 0.95,
			want: securityWant{isJailbreak: true, riskScore: 0.95, confidence: 0.95, recommend: "block", detectionType: "jailbreak"},
		},
		{
			name:        "includeReasoning adds a reasoning string for detected jailbreaks",
			isJailbreak: true, jailbreakType: "jailbreak", confidence: 0.95, riskScore: 0.95, includeReasoning: true,
			want: securityWant{isJailbreak: true, riskScore: 0.95, confidence: 0.95, recommend: "block", detectionType: "jailbreak", reasoning: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := buildSecurityResponse(tt.isJailbreak, tt.jailbreakType, tt.confidence, tt.riskScore, tt.includeReasoning, 42)
			assertSecurityResponse(t, resp, tt.want)
			if resp.ProcessingTimeMs != 42 {
				t.Errorf("ProcessingTimeMs = %d, want 42", resp.ProcessingTimeMs)
			}
		})
	}
}
