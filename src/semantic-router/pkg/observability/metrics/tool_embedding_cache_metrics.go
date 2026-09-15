package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ToolEmbeddingCacheHits counts request-supplied tool definitions whose
	// embedding was served from the in-process tool embedding memo.
	ToolEmbeddingCacheHits = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "vsr_tool_embedding_cache_hits_total",
			Help: "Total tool definitions whose embedding was served from the in-process tool embedding cache",
		},
	)

	// ToolEmbeddingCacheMisses counts unique tool embedding texts that had to be
	// computed because the memo did not hold them.
	ToolEmbeddingCacheMisses = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "vsr_tool_embedding_cache_misses_total",
			Help: "Total unique tool embedding texts that had to be computed on a tool embedding cache miss",
		},
	)

	// ToolEmbeddingBatchSize observes how many texts each tool-selection filter
	// pass actually embedded (unique tool misses plus the query).
	ToolEmbeddingBatchSize = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "vsr_tool_embedding_batch_size",
			Help:    "Number of texts embedded per tool-selection filter pass",
			Buckets: prometheus.ExponentialBuckets(1, 2, 8),
		},
	)
)

// RecordToolEmbeddingCacheHits records n tool embeddings served from the memo.
func RecordToolEmbeddingCacheHits(n int) {
	if n <= 0 {
		return
	}
	ToolEmbeddingCacheHits.Add(float64(n))
}

// RecordToolEmbeddingCacheMisses records n tool embedding texts that had to be computed.
func RecordToolEmbeddingCacheMisses(n int) {
	if n <= 0 {
		return
	}
	ToolEmbeddingCacheMisses.Add(float64(n))
}

// RecordToolEmbeddingBatchSize records how many texts were embedded in one pass.
func RecordToolEmbeddingBatchSize(n int) {
	if n < 0 {
		return
	}
	ToolEmbeddingBatchSize.Observe(float64(n))
}
