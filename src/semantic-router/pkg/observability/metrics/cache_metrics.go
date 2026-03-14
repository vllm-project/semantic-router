package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

var (
	// CacheOperationDuration tracks the duration of cache operations by backend and operation type
	CacheOperationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_cache_operation_duration_seconds",
			Help:    "The duration of cache operations in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
		},
		[]string{"backend", "operation"},
	)

	// CacheOperationTotal tracks the total number of cache operations by backend and operation type
	CacheOperationTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_cache_operations_total",
			Help: "The total number of cache operations",
		},
		[]string{"backend", "operation", "status"},
	)

	// CacheEntriesTotal tracks the total number of entries in the cache by backend
	CacheEntriesTotal = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_cache_entries_total",
			Help: "The total number of entries in the cache",
		},
		[]string{"backend"},
	)

	// CachePluginHits tracks cache hits by decision and plugin type
	CachePluginHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_cache_plugin_hits_total",
			Help: "The total number of cache hits by decision and plugin type",
		},
		[]string{"decision_name", "plugin_type"},
	)

	// CachePluginMisses tracks cache misses by decision and plugin type
	CachePluginMisses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_cache_plugin_misses_total",
			Help: "The total number of cache misses by decision and plugin type",
		},
		[]string{"decision_name", "plugin_type"},
	)

	// CacheWriteSkipped tracks cache write skips when response has personalized context (RAG, memory, PII, system prompt)
	CacheWriteSkipped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_cache_write_skipped_total",
			Help: "Total number of cache writes skipped because response contained personalized context (rag_context, memory_context, pii_detected, system_prompt_injected)",
		},
		[]string{"reason"},
	)
)

// RecordCacheOperation records a cache operation with duration and status
func RecordCacheOperation(backend, operation, status string, duration float64) {
	CacheOperationDuration.WithLabelValues(backend, operation).Observe(duration)
	CacheOperationTotal.WithLabelValues(backend, operation, status).Inc()
}

// UpdateCacheEntries updates the current number of cache entries for a backend
func UpdateCacheEntries(backend string, count int) {
	CacheEntriesTotal.WithLabelValues(backend).Set(float64(count))
}

// RecordCachePluginHit records a cache hit for a specific decision and plugin type
func RecordCachePluginHit(decisionName, pluginType string) {
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	if pluginType == "" {
		pluginType = "semantic-cache"
	}
	CachePluginHits.WithLabelValues(decisionName, pluginType).Inc()
}

// RecordCachePluginMiss records a cache miss for a specific decision and plugin type
func RecordCachePluginMiss(decisionName, pluginType string) {
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	if pluginType == "" {
		pluginType = "semantic-cache"
	}
	CachePluginMisses.WithLabelValues(decisionName, pluginType).Inc()
}

// Cache write skip reasons (canonical labels for CacheWriteSkipped metric)
const (
	CacheWriteSkipReasonRAGContext           = "rag_context"
	CacheWriteSkipReasonMemoryContext        = "memory_context"
	CacheWriteSkipReasonPIIDetected          = "pii_detected"
	CacheWriteSkipReasonSystemPromptInjected = "system_prompt_injected"
)

// RecordCacheWriteSkipped records that a cache write was skipped because the response has personalized context
func RecordCacheWriteSkipped(reason string) {
	if reason == "" {
		reason = consts.UnknownLabel
	}
	CacheWriteSkipped.WithLabelValues(reason).Inc()
}
