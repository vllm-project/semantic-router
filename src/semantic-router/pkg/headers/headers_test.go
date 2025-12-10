package headers

import (
	"testing"
)

func TestHeaderConstants(t *testing.T) {
	tests := []struct {
		name     string
		header   string
		expected string
	}{
		// Request headers
		{"RequestID", RequestID, "x-request-id"},
		{"GatewayDestinationEndpoint", GatewayDestinationEndpoint, "x-vsr-destination-endpoint"},
		{"SelectedModel", SelectedModel, "x-selected-model"},
		// VSR headers
		{"VSRSelectedCategory", VSRSelectedCategory, "x-vsr-selected-category"},
		{"VSRSelectedReasoning", VSRSelectedReasoning, "x-vsr-selected-reasoning"},
		{"VSRSelectedModel", VSRSelectedModel, "x-vsr-selected-model"},
		{"VSRInjectedSystemPrompt", VSRInjectedSystemPrompt, "x-vsr-injected-system-prompt"},
		{"VSRCacheHit", VSRCacheHit, "x-vsr-cache-hit"},
		// Security headers
		{"VSRPIIViolation", VSRPIIViolation, "x-vsr-pii-violation"},
		{"VSRJailbreakBlocked", VSRJailbreakBlocked, "x-vsr-jailbreak-blocked"},
		{"VSRJailbreakType", VSRJailbreakType, "x-vsr-jailbreak-type"},
		{"VSRJailbreakConfidence", VSRJailbreakConfidence, "x-vsr-jailbreak-confidence"},
		// Hallucination mitigation headers
		{"HallucinationDetected", HallucinationDetected, "x-vsr-hallucination-detected"},
		{"HallucinationSpans", HallucinationSpans, "x-vsr-hallucination-spans"},
		{"FactCheckNeeded", FactCheckNeeded, "x-vsr-fact-check-needed"},
		{"UnverifiedFactualResponse", UnverifiedFactualResponse, "x-vsr-unverified-factual-response"},
		{"VerificationContextMissing", VerificationContextMissing, "x-vsr-verification-context-missing"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.header != tt.expected {
				t.Errorf("Expected %s to be %q, got %q", tt.name, tt.expected, tt.header)
			}
		})
	}
}

func TestHallucinationMitigationHeaders(t *testing.T) {
	// Verify all hallucination mitigation headers follow the x-vsr- prefix convention
	hallucinationHeaders := []string{
		HallucinationDetected,
		HallucinationSpans,
		FactCheckNeeded,
		UnverifiedFactualResponse,
		VerificationContextMissing,
	}

	for _, h := range hallucinationHeaders {
		if len(h) < 6 || h[:6] != "x-vsr-" {
			t.Errorf("Header %q should start with 'x-vsr-' prefix", h)
		}
	}
}

func TestUnverifiedFactualResponseHeaders(t *testing.T) {
	// Test the specific headers added when a factual response cannot be verified
	if UnverifiedFactualResponse != "x-vsr-unverified-factual-response" {
		t.Errorf("UnverifiedFactualResponse header has wrong value: %s", UnverifiedFactualResponse)
	}

	if VerificationContextMissing != "x-vsr-verification-context-missing" {
		t.Errorf("VerificationContextMissing header has wrong value: %s", VerificationContextMissing)
	}

	// These headers should be used together
	if FactCheckNeeded != "x-vsr-fact-check-needed" {
		t.Errorf("FactCheckNeeded header has wrong value: %s", FactCheckNeeded)
	}
}
