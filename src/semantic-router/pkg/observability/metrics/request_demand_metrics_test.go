package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRequestDemandMetricContract(t *testing.T) {
	RecordRequestDemand("provider_bound", "fallback", 128, 64, 192, true)

	count, err := testutil.GatherAndCount(prometheus.DefaultGatherer, "llm_request_demand_tokens")
	if err != nil {
		t.Fatalf("gather request demand metric: %v", err)
	}
	if count != 3 {
		t.Fatalf("request demand metric series count = %d, want 3 components", count)
	}
}
