package headers

import (
	"os"
	"path/filepath"
	"strings"
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
		{"SelectedModel", SelectedModel, "x-selected-model"},
		{"VSRSkipProcessing", VSRSkipProcessing, "x-vsr-skip-processing"},
		// VSR headers
		{"VSRSelectedCategory", VSRSelectedCategory, "x-vsr-selected-category"},
		{"VSRSelectedReasoning", VSRSelectedReasoning, "x-vsr-selected-reasoning"},
		{"VSRSelectedModel", VSRSelectedModel, "x-vsr-selected-model"},
		{"VSRSessionPhase", VSRSessionPhase, "x-vsr-session-phase"},
		{"VSRInjectedSystemPrompt", VSRInjectedSystemPrompt, "x-vsr-injected-system-prompt"},
		{"VSRCacheHit", VSRCacheHit, "x-vsr-cache-hit"},
		{"VSRMatchedModality", VSRMatchedModality, "x-vsr-matched-modality"},
		{"VSRMatchedAuthz", VSRMatchedAuthz, "x-vsr-matched-authz"},
		{"VSRMatchedJailbreak", VSRMatchedJailbreak, "x-vsr-matched-jailbreak"},
		{"VSRMatchedPII", VSRMatchedPII, "x-vsr-matched-pii"},
		{"VSRMatchedReask", VSRMatchedReask, "x-vsr-matched-reask"},
		{"VSRMatchedEvent", VSRMatchedEvent, "x-vsr-matched-event"},
		{"VSRMatchedProjection", VSRMatchedProjection, "x-vsr-matched-projections"},
		// Consolidated response-warnings header (v0.4)
		{"VSRResponseWarnings", VSRResponseWarnings, "x-vsr-response-warnings"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.header != tt.expected {
				t.Errorf("Expected %s to be %q, got %q", tt.name, tt.expected, tt.header)
			}
		})
	}
}

func TestResponseWarningsCodes(t *testing.T) {
	// The v0.4 contract consolidates the hallucination / unverified-factual /
	// response-jailbreak warnings into x-vsr-response-warnings; these are the
	// codes carried in its value.
	cases := map[string]string{
		ResponseWarningHallucination:     "hallucination",
		ResponseWarningUnverifiedFactual: "unverified_factual",
		ResponseWarningJailbreak:         "response_jailbreak",
	}
	for got, want := range cases {
		if got != want {
			t.Errorf("response warning code = %q, want %q", got, want)
		}
	}
}

func TestVSRRoutingHeadersAreDocumented(t *testing.T) {
	docPath := filepath.Join("..", "..", "..", "..", "website", "docs", "troubleshooting", "vsr-headers.md")
	content, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", docPath, err)
	}
	doc := string(content)

	headers := []string{
		XSessionID,
		VSRSkipProcessing,
		VSRDebug,
		VSRClientProtocol,
		VSRUpstreamProtocol,
		VSRProtocolWarnings,
		RouterReplayID,
		VSRSelectedCategory,
		VSRSelectedDecision,
		VSRSelectedConfidence,
		VSRSelectedReasoning,
		VSRSelectedModality,
		VSRSelectedModel,
		VSRSessionPhase,
		VSRInjectedSystemPrompt,
		VSRMatchedKeywords,
		VSRMatchedEmbeddings,
		VSRMatchedDomains,
		VSRMatchedFactCheck,
		VSRMatchedUserFeedback,
		VSRMatchedReask,
		VSRMatchedPreference,
		VSRMatchedLanguage,
		VSRMatchedContext,
		VSRContextTokenCount,
		VSRMatchedStructure,
		VSRMatchedComplexity,
		VSRMatchedModality,
		VSRMatchedAuthz,
		VSRMatchedJailbreak,
		VSRMatchedPII,
		VSRMatchedKB,
		VSRMatchedConversation,
		VSRMatchedEvent,
		VSRMatchedProjection,
		VSRCacheHit,
		VSRFastResponse,
		VSRToolsStrategy,
		VSRToolsConfidence,
		VSRToolsLatencyMs,
	}

	for _, header := range headers {
		if !strings.Contains(doc, header) {
			t.Fatalf("%s should document public routing header %q", docPath, header)
		}
	}
}
