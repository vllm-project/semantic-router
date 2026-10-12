package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

func TestTopicContinuityEvaluationLabels(t *testing.T) {
	counter := topicContinuityEvaluations.WithLabelValues("change", "change_explicit_marker", "full", "false")
	before := testutil.ToFloat64(counter)
	RecordTopicContinuityEvaluation("change", "change_explicit_marker", "full", false)
	if got := testutil.ToFloat64(counter) - before; got != 1 {
		t.Fatalf("counter delta = %g, want 1", got)
	}
}

func TestTopicContinuityPreparationHistogram(t *testing.T) {
	var before dto.Metric
	if err := topicContinuityPreparedInputBytes.(prometheus.Metric).Write(&before); err != nil {
		t.Fatal(err)
	}
	RecordTopicContinuityPreparation(4096)
	var after dto.Metric
	if err := topicContinuityPreparedInputBytes.(prometheus.Metric).Write(&after); err != nil {
		t.Fatal(err)
	}
	if got := after.GetHistogram().GetSampleCount() - before.GetHistogram().GetSampleCount(); got != 1 {
		t.Fatalf("sample delta = %d, want 1", got)
	}
}
