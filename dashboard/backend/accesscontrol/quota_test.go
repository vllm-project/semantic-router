package accesscontrol

import (
	"testing"
	"time"
)

func TestQuotaMeterNeverReportsNegativeRemainingCapacity(t *testing.T) {
	reset := time.Date(2026, 8, 22, 12, 1, 0, 0, time.UTC)

	available := quotaMeter(100, 37, reset)
	if available.Limit != 100 || available.Used != 37 || available.Remaining != 63 || !available.ResetsAt.Equal(reset) {
		t.Fatalf("quotaMeter(available) = %#v", available)
	}

	exhausted := quotaMeter(100, 140, reset)
	if exhausted.Remaining != 0 {
		t.Fatalf("quotaMeter(exhausted).Remaining = %d, want 0", exhausted.Remaining)
	}
}

func TestQuotaErrorsUsePlainLanguageWithoutInternalBudgetIDs(t *testing.T) {
	testCases := map[string]string{
		"rpm":          "Request limit reached for this minute. Try again shortly.",
		"tpm":          "Token limit reached for this minute. Try again shortly.",
		"daily_tokens": "Daily token limit reached.",
	}
	for dimension, expected := range testCases {
		err := (&QuotaError{Dimension: dimension, BudgetID: "private-budget-id"}).UserMessage()
		if err != expected {
			t.Fatalf("UserMessage(%q) = %q, want %q", dimension, err, expected)
		}
	}
}
