package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEventClassifier_NoRules(t *testing.T) {
	ec := NewEventClassifier(nil)
	matches := ec.Classify("payment_failed critical TXN_DECLINE urgent")
	if len(matches) != 0 {
		t.Fatalf("expected no matches with no rules, got %d", len(matches))
	}
}

func TestEventClassifier_EventTypeMatch(t *testing.T) {
	rules := []config.EventRule{
		{Name: "payment_rule", EventTypes: []string{"payment_failed"}},
	}
	ec := NewEventClassifier(rules)

	matches := ec.Classify("User reported a payment_failed error in checkout")
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	if matches[0].RuleName != "payment_rule" {
		t.Errorf("unexpected rule name: %s", matches[0].RuleName)
	}
	if matches[0].Confidence < 0.5 {
		t.Errorf("confidence too low: %f", matches[0].Confidence)
	}
}

func TestEventClassifier_NoMatchOnUnrelatedText(t *testing.T) {
	rules := []config.EventRule{
		{Name: "payment_rule", EventTypes: []string{"payment_failed"}},
	}
	ec := NewEventClassifier(rules)
	matches := ec.Classify("What is the weather like today?")
	if len(matches) != 0 {
		t.Fatalf("expected no matches, got %d", len(matches))
	}
}

func TestEventClassifier_SeverityMatch(t *testing.T) {
	rules := []config.EventRule{
		{Name: "critical_rule", Severities: []string{"critical"}},
	}
	ec := NewEventClassifier(rules)

	matches := ec.Classify("CRITICAL: database connection lost")
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	if matches[0].MatchedSeverity != "critical" {
		t.Errorf("expected severity 'critical', got %q", matches[0].MatchedSeverity)
	}
}

func TestEventClassifier_TemporalMatch(t *testing.T) {
	rules := []config.EventRule{
		{Name: "urgent_rule", Temporal: true},
	}
	ec := NewEventClassifier(rules)

	matches := ec.Classify("Please handle this urgent request immediately")
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	if !matches[0].TemporalMatch {
		t.Error("expected TemporalMatch=true")
	}
}

func TestEventClassifier_ActionCodeMatch(t *testing.T) {
	rules := []config.EventRule{
		{Name: "txn_rule", ActionCodes: []string{"TXN_DECLINE"}},
	}
	ec := NewEventClassifier(rules)

	matches := ec.Classify("Error code TXN_DECLINE received from payment gateway")
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
}

func TestEventClassifier_MultiCriteriaHighConfidence(t *testing.T) {
	rules := []config.EventRule{
		{
			Name:        "full_rule",
			EventTypes:  []string{"auth_error"},
			Severities:  []string{"high"},
			ActionCodes: []string{"AUTH_FAIL"},
			Temporal:    true,
		},
	}
	ec := NewEventClassifier(rules)

	text := "auth_error detected: AUTH_FAIL code, high severity, urgent action required"
	matches := ec.Classify(text)
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	// All 4 criteria matched → confidence should be 1.0
	if matches[0].Confidence != 1.0 {
		t.Errorf("expected confidence 1.0 with all criteria matched, got %f", matches[0].Confidence)
	}
}

func TestEventClassifier_PartialCriteriaLowerConfidence(t *testing.T) {
	rules := []config.EventRule{
		{
			Name:       "partial_rule",
			EventTypes: []string{"payment_failed"},
			Severities: []string{"critical"},
		},
	}
	ec := NewEventClassifier(rules)

	// Only event type matches, not severity
	matches := ec.Classify("payment_failed occurred")
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	// 1 of 2 criteria → confidence = 0.5 + 0.5*(1/2) = 0.75
	if matches[0].Confidence != 0.75 {
		t.Errorf("expected confidence 0.75, got %f", matches[0].Confidence)
	}
}

func TestEventClassifier_MultipleRulesMultipleMatches(t *testing.T) {
	rules := []config.EventRule{
		{Name: "rule_a", EventTypes: []string{"payment_failed"}},
		{Name: "rule_b", Severities: []string{"critical"}},
	}
	ec := NewEventClassifier(rules)

	matches := ec.Classify("payment_failed with critical severity")
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
}

func TestEventClassifier_CaseInsensitiveMatching(t *testing.T) {
	rules := []config.EventRule{
		{Name: "rule", EventTypes: []string{"Payment_Failed"}},
	}
	ec := NewEventClassifier(rules)

	if matches := ec.Classify("PAYMENT_FAILED in production"); len(matches) != 1 {
		t.Fatalf("expected case-insensitive match, got %d matches", len(matches))
	}
}

func TestEventClassifier_EmptyRuleNeverMatches(t *testing.T) {
	// A rule with no criteria configured should never match anything.
	rules := []config.EventRule{
		{Name: "empty_rule"},
	}
	ec := NewEventClassifier(rules)
	matches := ec.Classify("critical payment_failed TXN_DECLINE urgent")
	if len(matches) != 0 {
		t.Fatalf("empty rule should not match, got %d matches", len(matches))
	}
}
