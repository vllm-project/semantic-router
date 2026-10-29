package classification

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Balance's legal route needs this keyword when the decision model files a
// contract-risk request under business rather than law (#5592).
func TestBalanceLegalRiskMarkersMatchTheEnglishLegalRiskRequest(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "balance", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	classifier, err := NewKeywordClassifier(cfg.KeywordRules)
	if err != nil {
		t.Fatal(err)
	}
	matches, err := classifier.MatchAll("Assess the legal risk in this agreement by analyzing indemnification, limitation of liability, and the compliance duties each party assumes.")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(matches, func(m KeywordRuleMatch) bool { return m.RuleName == "legal_risk_markers" }) {
		t.Fatalf("legal_risk_markers did not match, got %+v", matches)
	}
}
