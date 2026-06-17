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
		// Hallucination mitigation headers
		{"HallucinationDetected", HallucinationDetected, "x-vsr-hallucination-detected"},
		{"HallucinationSpans", HallucinationSpans, "x-vsr-hallucination-spans"},
		{"FactCheckNeeded", FactCheckNeeded, "x-vsr-fact-check-needed"},
		{"UnverifiedFactualResponse", UnverifiedFactualResponse, "x-vsr-unverified-factual-response"},
		{"VerificationContextMissing", VerificationContextMissing, "x-vsr-verification-context-missing"},
		// Response jailbreak detection headers
		{"ResponseJailbreakDetected", ResponseJailbreakDetected, "x-vsr-response-jailbreak-detected"},
		{"ResponseJailbreakType", ResponseJailbreakType, "x-vsr-response-jailbreak-type"},
		{"ResponseJailbreakConfidence", ResponseJailbreakConfidence, "x-vsr-response-jailbreak-confidence"},
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
