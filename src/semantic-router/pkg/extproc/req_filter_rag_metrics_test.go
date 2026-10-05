package extproc

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestFinalizeRAGRetrievalDistinguishesZeroFromUnreportedCount(t *testing.T) {
	metrics.RAGResultCount.Reset()

	router := &OpenAIRouter{}
	ragConfig := &config.RAGPluginConfig{Backend: "vectorstore"}
	tracer := noop.NewTracerProvider().Tracer("rag-result-count-test")

	unreported := &RequestContext{}
	_, unreportedSpan := tracer.Start(context.Background(), "unreported")
	if err := router.finalizeRAGRetrieval(unreported, unreportedSpan, ragConfig, "decision", "", 0); err != nil {
		t.Fatal(err)
	}
	unreportedSpan.End()
	if got := testutil.CollectAndCount(metrics.RAGResultCount); got != 0 {
		t.Fatalf("unreported result count created %d metric series, want 0", got)
	}

	reported := &RequestContext{RAGResultCountReported: true}
	_, reportedSpan := tracer.Start(context.Background(), "reported-zero")
	if err := router.finalizeRAGRetrieval(reported, reportedSpan, ragConfig, "decision", "", 0); err != nil {
		t.Fatal(err)
	}
	reportedSpan.End()

	observer := metrics.RAGResultCount.WithLabelValues("vectorstore", requestDecisionStateKey(reported))
	metric, ok := observer.(prometheus.Metric)
	if !ok {
		t.Fatalf("result-count observer has type %T, want prometheus.Metric", observer)
	}
	var value dto.Metric
	if err := metric.Write(&value); err != nil {
		t.Fatal(err)
	}
	if got := value.GetHistogram().GetSampleCount(); got != 1 {
		t.Fatalf("reported zero sample count = %d, want 1", got)
	}
	if got := value.GetHistogram().GetSampleSum(); got != 0 {
		t.Fatalf("reported zero sample sum = %v, want 0", got)
	}
}
