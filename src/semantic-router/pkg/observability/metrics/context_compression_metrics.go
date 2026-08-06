package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	contextCompressionTokensBefore = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_tokens_before",
			Help:    "Estimated tool-result tokens before route-local compression.",
			Buckets: []float64{128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000},
		},
		[]string{"format"},
	)
	contextCompressionTokensAfter = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_tokens_after",
			Help:    "Estimated tool-result tokens after route-local compression.",
			Buckets: []float64{64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000},
		},
		[]string{"format"},
	)
	contextCompressionRatio = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_compression_ratio",
			Help:    "Compressed-to-original estimated token ratio.",
			Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1},
		},
		[]string{"format"},
	)
)

// RecordContextCompression records bounded, content-free compression
// diagnostics. Format is a low-cardinality value: text, json, or mixed.
func RecordContextCompression(before int, after int, format string) {
	if before <= 0 || after < 0 {
		return
	}
	if format == "" {
		format = "text"
	}
	contextCompressionTokensBefore.WithLabelValues(format).Observe(float64(before))
	contextCompressionTokensAfter.WithLabelValues(format).Observe(float64(after))
	contextCompressionRatio.WithLabelValues(format).Observe(float64(after) / float64(before))
}
