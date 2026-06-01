package classification

import (
	"testing"
)

// Security regression: prior USER turns must be scannable by history-aware
// negative signals (PII/jailbreak), else secrets from an earlier turn leak to
// external providers on a later benign turn. See issue #1961.
func TestHistoryForHistoryAwareSignals_IncludesPriorUserTurns(t *testing.T) {
	priorUserMessages := []string{"my prod key is sk-SECRET-123"}
	nonUserMessages := []string{"You are a helpful assistant.", "Sure, here is how rotation works."}

	got := historyForHistoryAwareSignals(priorUserMessages, nonUserMessages)

	// Must contain every non-user message (existing behavior preserved).
	for _, want := range nonUserMessages {
		if !containsString(got, want) {
			t.Errorf("expected history to preserve non-user message %q, got %v", want, got)
		}
	}

	// Must ALSO contain the prior user message carrying the secret (the fix).
	for _, want := range priorUserMessages {
		if !containsString(got, want) {
			t.Errorf("SECURITY: prior user message %q must be scanned by history-aware signals, got %v", want, got)
		}
	}
}

func TestHistoryForHistoryAwareSignals_NoDuplicatesAndDropsEmpty(t *testing.T) {
	priorUserMessages := []string{"shared", "", "user-only"}
	nonUserMessages := []string{"shared", "assistant-only", ""}

	got := historyForHistoryAwareSignals(priorUserMessages, nonUserMessages)

	if containsString(got, "") {
		t.Errorf("expected empty strings to be dropped, got %v", got)
	}
	if countString(got, "shared") != 1 {
		t.Errorf("expected %q to appear exactly once, got %v", "shared", got)
	}
	for _, want := range []string{"user-only", "assistant-only", "shared"} {
		if !containsString(got, want) {
			t.Errorf("expected merged history to contain %q, got %v", want, got)
		}
	}
}

func countString(s []string, target string) int {
	n := 0
	for _, v := range s {
		if v == target {
			n++
		}
	}
	return n
}
