package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// The remote-signal metric families must be registered and gatherable under
// their documented names; a renamed or unregistered family breaks every
// dashboard that reads it without any test failing elsewhere.
func TestRemoteSignalMetricContract(t *testing.T) {
	RecordComplexityVerdict("needs_reasoning", "hard", "remote_score")
	RecordComplexityEvaluationFailure("remote_score")
	RecordRemoteConnectorRequest("http_classify_score", RemoteConnectorOutcomeSuccess, 0.02)
	RecordRemoteConnectorRetry("http_classify_score")

	for _, metricName := range []string{
		"llm_complexity_verdict_total",
		"llm_complexity_evaluation_failures_total",
		"llm_remote_connector_request_duration_seconds",
		"llm_remote_connector_requests_total",
		"llm_remote_connector_retries_total",
	} {
		count, err := testutil.GatherAndCount(prometheus.DefaultGatherer, metricName)
		if err != nil {
			t.Fatalf("gather %s: %v", metricName, err)
		}
		if count == 0 {
			t.Fatalf("expected Prometheus metric family %q to be gathered", metricName)
		}
	}
}

// Empty labels are a cardinality and readability hazard; they must land on
// the shared unknown label rather than an empty string.
func TestRemoteSignalMetricsNormalizeEmptyLabels(t *testing.T) {
	before := testutil.ToFloat64(ComplexityVerdictTotal.WithLabelValues("unknown", "unknown", "unknown"))
	RecordComplexityVerdict("", "", "")
	after := testutil.ToFloat64(ComplexityVerdictTotal.WithLabelValues("unknown", "unknown", "unknown"))
	if after != before+1 {
		t.Fatalf("empty labels should count under unknown: before=%v after=%v", before, after)
	}
}
