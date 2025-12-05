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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.header != tt.expected {
				t.Errorf("Expected %s to be %q, got %q", tt.name, tt.expected, tt.header)
			}
		})
	}
}
