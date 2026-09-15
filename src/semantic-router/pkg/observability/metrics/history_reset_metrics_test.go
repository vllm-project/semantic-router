package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestHistoryResetMetricsRecordCommittedOutcomesOnly(t *testing.T) {
	RecordHistoryResetEvaluation("reset-decision", "applied", "applied", 2, 4)
	RecordHistoryResetEvaluation("reset-decision", "skipped", "evidence_continuation", 0, 0)
	RecordHistoryResetRecovery("stored")

	if testutil.CollectAndCount(historyResetEvaluations, "llm_history_reset_evaluations_total") == 0 {
		t.Fatal("evaluations metric was not registered")
	}
	if testutil.CollectAndCount(historyResetRemovedTurns, "llm_history_reset_removed_turns") == 0 {
		t.Fatal("removed-turns metric was not registered")
	}
	if testutil.CollectAndCount(historyResetRemovedMessages, "llm_history_reset_removed_messages") == 0 {
		t.Fatal("removed-messages metric was not registered")
	}
	if testutil.CollectAndCount(historyResetRecovery, "llm_history_reset_recovery_total") == 0 {
		t.Fatal("recovery metric was not registered")
	}
	if got := testutil.ToFloat64(
		historyResetEvaluations.WithLabelValues("reset-decision", "skipped", "evidence_continuation"),
	); got != 1 {
		t.Fatalf("skipped evaluations = %v, want 1", got)
	}

	// Only committed removals reach the histograms.
	if got := testutil.CollectAndCount(historyResetRemovedTurns, "llm_history_reset_removed_turns"); got != 1 {
		t.Fatalf("removed-turn series = %d, want 1", got)
	}
}

func TestHistoryResetRecoveryIgnoresAnEmptyStatus(t *testing.T) {
	before := testutil.CollectAndCount(historyResetRecovery, "llm_history_reset_recovery_total")
	RecordHistoryResetRecovery("")
	if after := testutil.CollectAndCount(historyResetRecovery, "llm_history_reset_recovery_total"); after != before {
		t.Fatal("an empty recovery status must not create a series")
	}
}
