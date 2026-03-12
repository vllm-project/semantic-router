package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// ContextTokenCount tracks the distribution of input token counts for context-based routing
var ContextTokenCount = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "llm_context_token_count",
		Help:    "Distribution of input token counts for context-based routing",
		Buckets: []float64{100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000},
	},
	[]string{"model", "context_level"},
)

// RecordContextTokenCount records the input token count with context level
func RecordContextTokenCount(model string, tokenCount int, contextLevel string) {
	if model == "" {
		model = consts.UnknownLabel
	}
	if contextLevel == "" {
		contextLevel = consts.UnknownLabel
	}
	ContextTokenCount.WithLabelValues(model, contextLevel).Observe(float64(tokenCount))
}
