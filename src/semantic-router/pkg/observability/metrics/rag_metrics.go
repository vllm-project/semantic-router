package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RAGRetrievalAttempts tracks RAG retrieval attempts
	RAGRetrievalAttempts = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_retrieval_attempts_total",
			Help: "Total number of RAG retrieval attempts",
		},
		[]string{"backend", "decision", "status"},
	)

	// RAGRetrievalLatency tracks RAG retrieval latency
	RAGRetrievalLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_retrieval_latency_seconds",
			Help:    "RAG retrieval latency in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
		},
		[]string{"backend", "decision"},
	)

	// RAGSimilarityScore tracks RAG similarity score
	RAGSimilarityScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "rag_similarity_score",
			Help: "Average similarity score of retrieved documents",
		},
		[]string{"backend", "decision"},
	)

	// RAGContextLength tracks RAG context length
	RAGContextLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_context_length_chars",
			Help:    "Length of retrieved context in characters",
			Buckets: []float64{100, 500, 1000, 2000, 5000, 10000, 20000},
		},
		[]string{"backend", "decision"},
	)

	// RAGCacheHits tracks RAG cache hits
	RAGCacheHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_cache_hits_total",
			Help: "Total number of RAG cache hits",
		},
		[]string{"backend"},
	)

	// RAGCacheMisses tracks RAG cache misses
	RAGCacheMisses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_cache_misses_total",
			Help: "Total number of RAG cache misses",
		},
		[]string{"backend"},
	)
)

// RecordRAGRetrieval records a RAG retrieval attempt
func RecordRAGRetrieval(backend string, decision string, status string, latency float64) {
	RAGRetrievalAttempts.WithLabelValues(backend, decision, status).Inc()
	if latency > 0 {
		RAGRetrievalLatency.WithLabelValues(backend, decision).Observe(latency)
	}
}

// RecordRAGSimilarityScore records the similarity score from RAG retrieval.
// We intentionally use a Gauge here to expose only the latest similarity score
// per (backend, decision) label combination, rather than tracking the full
// distribution of scores over time (which would require a Histogram).
func RecordRAGSimilarityScore(backend string, decision string, score float32) {
	RAGSimilarityScore.WithLabelValues(backend, decision).Set(float64(score))
}

// RecordRAGContextLength records the length of retrieved context
func RecordRAGContextLength(backend string, decision string, length int) {
	RAGContextLength.WithLabelValues(backend, decision).Observe(float64(length))
}

// RecordRAGCacheHit records a RAG cache hit
func RecordRAGCacheHit(backend string) {
	RAGCacheHits.WithLabelValues(backend).Inc()
}

// RecordRAGCacheMiss records a RAG cache miss
func RecordRAGCacheMiss(backend string) {
	RAGCacheMisses.WithLabelValues(backend).Inc()
}
