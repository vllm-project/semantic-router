package metrics

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
