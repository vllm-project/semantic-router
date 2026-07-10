package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
)

func TestCoreObservabilityMetricContract(t *testing.T) {
	model := "contract-test-model"

	RecordModelCompletionLatency(model, 0.42)
	RecordModelTTFT(model, 0.12)
	RecordModelTPOT(model, 0.01)
	RecordModelRoutingLatency(0.004)

	inflight.Reset()
	token := inflight.Begin(model)
	defer func() {
		inflight.End(model, token)
		inflight.Reset()
	}()

	for _, metricName := range []string{
		"llm_model_completion_latency_seconds",
		"llm_model_ttft_seconds",
		"llm_model_tpot_seconds",
		"llm_model_routing_latency_seconds",
		"llm_model_inflight_requests",
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
