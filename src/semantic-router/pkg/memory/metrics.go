package memory

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// =============================================================================
// Memory Metrics - Prometheus metrics for agentic memory operations
// =============================================================================

var (
	// MemoryRetrievalLatency tracks the latency of memory retrieval operations
	MemoryRetrievalLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_memory_retrieval_latency_seconds",
			Help:    "The duration of memory retrieval operations in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
		},
		[]string{"backend", "operation"},
	)

	// MemoryRetrievalCount tracks the total number of memory retrievals
	MemoryRetrievalCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_memory_retrieval_total",
			Help: "The total number of memory retrieval operations",
		},
		[]string{"backend", "status", "user_id"},
	)

	// MemoryRetrievalResults tracks the number of memories returned per query
	MemoryRetrievalResults = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_memory_retrieval_results",
			Help:    "The number of memories returned per retrieval query",
			Buckets: []float64{0, 1, 2, 3, 5, 10, 20, 50, 100},
		},
		[]string{"backend"},
	)

	// MemoryExtractionCount tracks the total number of memory extraction operations
	MemoryExtractionCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_memory_extraction_total",
			Help: "The total number of memory extraction operations",
		},
		[]string{"status"},
	)

	// MemoryExtractionLatency tracks the latency of LLM-based memory extraction
	MemoryExtractionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_memory_extraction_latency_seconds",
			Help:    "The duration of LLM-based memory extraction in seconds",
			Buckets: []float64{0.1, 0.5, 1, 2.5, 5, 10, 30, 60},
		},
		[]string{"status"},
	)

	// MemoryExtractionFactsCount tracks the number of facts extracted per batch
	MemoryExtractionFactsCount = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_memory_extraction_facts_count",
			Help:    "The number of facts extracted per extraction batch",
			Buckets: []float64{0, 1, 2, 3, 5, 10, 20, 50},
		},
		[]string{"type"},
	)

	// MemoryStoreOperations tracks memory store, update, and delete operations
	MemoryStoreOperations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_memory_store_operations_total",
			Help: "The total number of memory store operations",
		},
		[]string{"backend", "operation", "status"},
	)

	// MemoryStoreSize tracks the total number of memories stored
	MemoryStoreSize = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_memory_store_size",
			Help: "The total number of memories stored (per user if feasible)",
		},
		[]string{"backend", "user_id"},
	)
)

// =============================================================================
// Memory Metrics Recording Functions
// =============================================================================

// RecordMemoryRetrieval records a memory retrieval operation with duration and status
func RecordMemoryRetrieval(backend, operation, status, userID string, duration float64, resultCount int) {
	if backend == "" {
		backend = consts.UnknownLabel
	}
	if operation == "" {
		operation = "retrieve"
	}
	if status == "" {
		status = "success"
	}
	if userID == "" {
		userID = consts.UnknownLabel
	}

	MemoryRetrievalLatency.WithLabelValues(backend, operation).Observe(duration)
	MemoryRetrievalCount.WithLabelValues(backend, status, userID).Inc()

	if resultCount >= 0 {
		MemoryRetrievalResults.WithLabelValues(backend).Observe(float64(resultCount))
	}
}

// RecordMemoryExtraction records a memory extraction operation with duration and status
func RecordMemoryExtraction(status string, duration float64, factsCount int, factType string) {
	if status == "" {
		status = "success"
	}

	MemoryExtractionCount.WithLabelValues(status).Inc()
	MemoryExtractionLatency.WithLabelValues(status).Observe(duration)

	if factsCount >= 0 && factType != "" {
		MemoryExtractionFactsCount.WithLabelValues(factType).Observe(float64(factsCount))
	}
}

// RecordMemoryStoreOperation records a memory store operation
func RecordMemoryStoreOperation(backend, operation, status string, duration float64) {
	if backend == "" {
		backend = consts.UnknownLabel
	}
	if operation == "" {
		operation = "store"
	}
	if status == "" {
		status = "success"
	}

	MemoryStoreOperations.WithLabelValues(backend, operation, status).Inc()

	// Optional: record latency if operation-specific latency metrics are added
	if duration >= 0 {
		MemoryRetrievalLatency.WithLabelValues(backend, operation).Observe(duration)
	}
}

// UpdateMemoryStoreSize updates the total number of memories stored
func UpdateMemoryStoreSize(backend, userID string, count int) {
	if backend == "" {
		backend = consts.UnknownLabel
	}
	if userID == "" {
		userID = consts.UnknownLabel
	}

	MemoryStoreSize.WithLabelValues(backend, userID).Set(float64(count))
}
