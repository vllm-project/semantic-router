package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRecordModelPromptCacheUsage(t *testing.T) {
	model := "provider-cache-metrics-test"
	RecordModelPromptCacheUsage(model, 12, 5, "authoritative")
	RecordModelPromptCacheUsage(model, 3, 0, "partial")
	RecordModelPromptCacheUsage(model, 100, 100, "derived")

	if got := testutil.ToFloat64(ModelPromptCacheReadTokens.WithLabelValues(model)); got != 15 {
		t.Fatalf("read tokens = %v, want 15", got)
	}
	if got := testutil.ToFloat64(ModelPromptCacheWriteTokens.WithLabelValues(model)); got != 5 {
		t.Fatalf("write tokens = %v, want 5", got)
	}
	if got := testutil.ToFloat64(ModelPromptCacheUsageRecords.WithLabelValues(model, "authoritative")); got != 1 {
		t.Fatalf("authoritative records = %v, want 1", got)
	}
	if got := testutil.ToFloat64(ModelPromptCacheUsageRecords.WithLabelValues(model, "partial")); got != 1 {
		t.Fatalf("partial records = %v, want 1", got)
	}
	if got := testutil.ToFloat64(ModelPromptCacheUsageRecords.WithLabelValues(model, "derived")); got != 1 {
		t.Fatalf("derived records = %v, want 1", got)
	}
}
