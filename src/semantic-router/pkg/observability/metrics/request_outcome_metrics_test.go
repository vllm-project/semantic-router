package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRequestOutcomeMetricsAreBoundedAndCoverLongResponses(t *testing.T) {
	before := testutil.ToFloat64(requestOutcomes.WithLabelValues("other", "error"))
	RecordRequestOutcome("private-path", "raw-provider-error", 600)
	if testutil.ToFloat64(requestOutcomes.WithLabelValues("other", "error")) != before+1 {
		t.Fatal("unbounded terminal labels")
	}
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range families {
		if f.GetName() != "llm_request_duration_seconds" {
			continue
		}
		for _, m := range f.Metric {
			buckets := m.GetHistogram().Bucket
			if len(buckets) == 0 || buckets[len(buckets)-1].GetUpperBound() != 1800 {
				t.Fatal("long response durations lack finite buckets")
			}
			for _, label := range m.Label {
				if label.GetValue() == "private-path" || label.GetValue() == "raw-provider-error" {
					t.Fatal("private metric label")
				}
			}
		}
		return
	}
	t.Fatal("request duration was not emitted")
}
