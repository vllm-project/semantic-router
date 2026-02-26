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

	// --- MINJA defense metrics (arXiv:2503.03704) ---

	// MemoryMinjaFilterTotal tracks MINJA filter outcomes on the read path.
	MemoryMinjaFilterTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_memory_minja_filter_total",
			Help: "MINJA defense filter outcomes (result=passed|filtered)",
		},
		[]string{"result"},
	)

	// MemorySharedFilteredTotal tracks non-owner memories filtered by the MINJA filter.
	MemorySharedFilteredTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_memory_shared_filtered_total",
			Help: "Non-owner memories filtered by MINJA defense (reason=low_similarity|shared_cap|creator_cap)",
		},
		[]string{"reason"},
	)

	// MemorySharedAcceptedTotal tracks non-owner memories that passed the MINJA filter.
	MemorySharedAcceptedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_memory_shared_accepted_total",
			Help: "Non-owner memories that passed the MINJA defense filter",
		},
	)

	// MemoryRateLimitBlockedTotal tracks memory creation attempts blocked by the rate limiter.
	MemoryRateLimitBlockedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_memory_rate_limit_blocked_total",
			Help: "Memory creation attempts blocked by per-user rate limiter (anti-PSS defense)",
		},
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

// RecordMinjaFilter records a MINJA filter outcome on the read path.
func RecordMinjaFilter(passed bool) {
	if passed {
		MemoryMinjaFilterTotal.WithLabelValues("passed").Inc()
	} else {
		MemoryMinjaFilterTotal.WithLabelValues("filtered").Inc()
	}
}

// RecordSharedMemoryFiltered records a non-owner memory being filtered, with a reason.
func RecordSharedMemoryFiltered(reason string) {
	MemorySharedFilteredTotal.WithLabelValues(reason).Inc()
}

// RecordSharedMemoryAccepted records a non-owner memory passing the gate.
func RecordSharedMemoryAccepted() {
	MemorySharedAcceptedTotal.Inc()
}

// RecordRateLimitBlocked records a memory creation blocked by the per-user rate limiter.
func RecordRateLimitBlocked() {
	MemoryRateLimitBlockedTotal.Inc()
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
