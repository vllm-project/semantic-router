package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestContextDedupMetricsRecordCommittedOutcomesOnly(t *testing.T) {
	RecordContextDedupEvaluation("dedup-decision", "applied", "applied", 2, 4, 512)
	RecordContextDedupEvaluation("dedup-decision", "skipped", "no_duplicates", 0, 0, 0)
	RecordContextDedupEvaluation("dedup-decision", "failed", "history_limit_exceeded", 3, 0, 0)

	if testutil.CollectAndCount(contextDedupEvaluations, "llm_context_dedup_evaluations_total") == 0 {
		t.Fatal("evaluations metric was not registered")
	}
	if got := testutil.ToFloat64(
		contextDedupEvaluations.WithLabelValues("dedup-decision", "skipped", "no_duplicates"),
	); got != 1 {
		t.Fatalf("skipped evaluations = %v, want 1", got)
	}
	// Only committed removals reach the histograms: the failed evaluation
	// carried a turn count but no removed messages.
	if got := testutil.CollectAndCount(contextDedupRemovedTurns, "llm_context_dedup_removed_turns"); got != 1 {
		t.Fatalf("removed-turn series = %d, want 1", got)
	}
	if got := testutil.CollectAndCount(contextDedupRemovedMessages, "llm_context_dedup_removed_messages"); got != 1 {
		t.Fatalf("removed-message series = %d, want 1", got)
	}
	if got := testutil.CollectAndCount(contextDedupRemovedTextBytes, "llm_context_dedup_removed_text_bytes"); got != 1 {
		t.Fatalf("removed-bytes series = %d, want 1", got)
	}
}
