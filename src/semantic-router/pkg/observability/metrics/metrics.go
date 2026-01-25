package metrics

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// Minimal fallback bucket configurations - used only when configuration is completely missing
var (
	// Basic fallback buckets for emergency use when config.yaml is unavailable
	FallbackDurationBuckets = []float64{0.001, 0.01, 0.1, 1, 10}
	FallbackSizeBuckets     = []float64{1, 10, 100}
)

// Configuration constants
const (
	DefaultSampleRate = 1.0
	MinSampleRate     = 0.0
	MaxSampleRate     = 1.0
)

// BatchMetricsConfig represents configuration for batch classification metrics
type BatchMetricsConfig struct {
	SampleRate      float64                       `yaml:"sample_rate"`
	DurationBuckets []float64                     `yaml:"duration_buckets"`
	SizeBuckets     []float64                     `yaml:"size_buckets"`
	BatchSizeRanges []config.BatchSizeRangeConfig `yaml:"batch_size_ranges"`

	// Boolean fields grouped together to minimize padding
	Enabled                   bool `yaml:"enabled"`
	DetailedGoroutineTracking bool `yaml:"detailed_goroutine_tracking"`
	HighResolutionTiming      bool `yaml:"high_resolution_timing"`
}

// Global configuration for batch metrics
var (
	batchMetricsConfig BatchMetricsConfig
	configMutex        sync.RWMutex
	metricsInitOnce    sync.Once
)

// SetBatchMetricsConfig sets the configuration for batch classification metrics
func SetBatchMetricsConfig(config BatchMetricsConfig) {
	configMutex.Lock()
	defer configMutex.Unlock()

	batchMetricsConfig = config

	// Set default values if not provided
	if batchMetricsConfig.SampleRate <= MinSampleRate {
		batchMetricsConfig.SampleRate = DefaultSampleRate
	}
	if len(batchMetricsConfig.DurationBuckets) == 0 {
		batchMetricsConfig.DurationBuckets = FallbackDurationBuckets
	}
	if len(batchMetricsConfig.SizeBuckets) == 0 {
		batchMetricsConfig.SizeBuckets = FallbackSizeBuckets
	}

	// Initialize metrics with the configuration
	InitializeBatchMetrics(batchMetricsConfig)
}

// GetBatchMetricsConfig returns the current batch metrics configuration
func GetBatchMetricsConfig() BatchMetricsConfig {
	configMutex.RLock()
	defer configMutex.RUnlock()
	return batchMetricsConfig
}

// shouldCollectMetric determines if a metric should be collected based on sample rate
func shouldCollectMetric() bool {
	configMutex.RLock()
	sampleRate := batchMetricsConfig.SampleRate
	enabled := batchMetricsConfig.Enabled
	configMutex.RUnlock()

	if !enabled {
		return false
	}

	if sampleRate >= 1.0 {
		return true
	}

	return rand.Float64() < sampleRate
}

var (
	// ModelRequests tracks the number of requests made to each model
	ModelRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_requests_total",
			Help: "The total number of requests made to each LLM model",
		},
		[]string{"model"},
	)

	// RequestErrorsTotal tracks request errors categorized by reason
	RequestErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_request_errors_total",
			Help: "The total number of request errors categorized by reason (e.g., timeout, upstream_5xx, pii_policy_denied, jailbreak_block, parse_error, serialization_error, cancellation)",
		},
		[]string{"model", "reason"},
	)

	// ModelCost tracks the total cost attributed to each model by currency
	ModelCost = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_cost_total",
			Help: "The total cost attributed to each LLM model, labeled by currency",
		},
		[]string{"model", "currency"},
	)

	// ModelTokens tracks the number of tokens used by each model
	ModelTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_tokens_total",
			Help: "The total number of tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// ModelPromptTokens tracks the number of prompt tokens used by each model
	ModelPromptTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_prompt_tokens_total",
			Help: "The total number of prompt tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// ModelCompletionTokens tracks the number of completion tokens used by each model
	ModelCompletionTokens = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_completion_tokens_total",
			Help: "The total number of completion tokens used by each LLM model",
		},
		[]string{"model"},
	)

	// PromptTokensPerRequest tracks the distribution of prompt tokens per request by model
	PromptTokensPerRequest = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_prompt_tokens_per_request",
			Help:    "Distribution of prompt tokens per request by model",
			Buckets: []float64{0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384},
		},
		[]string{"model"},
	)

	// CompletionTokensPerRequest tracks the distribution of completion tokens per request by model
	CompletionTokensPerRequest = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_completion_tokens_per_request",
			Help:    "Distribution of completion tokens per request by model",
			Buckets: []float64{0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384},
		},
		[]string{"model"},
	)

	// ModelRoutingModifications tracks when a model is changed from one to another
	ModelRoutingModifications = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_routing_modifications_total",
			Help: "The total number of times a request was rerouted from source model to target model",
		},
		[]string{"source_model", "target_model"},
	)

	// RoutingReasonCodes tracks routing decisions by reason_code and model
	RoutingReasonCodes = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_routing_reason_codes_total",
			Help: "The total number of routing decisions by reason code and model",
		},
		[]string{"reason_code", "model"},
	)

	// ModelCompletionLatency tracks the latency of completions by model
	ModelCompletionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_model_completion_latency_seconds",
			Help:    "The latency of LLM model completions in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// ModelTTFT tracks time to first token by model
	ModelTTFT = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_model_ttft_seconds",
			Help:    "Time to first token for LLM model responses in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// ModelTPOT tracks time per output token by model
	ModelTPOT = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_model_tpot_seconds",
			Help:    "Time per output token (completion latency / completion tokens) for LLM model responses in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// ModelRoutingLatency tracks the latency of model routing
	ModelRoutingLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_model_routing_latency_seconds",
			Help:    "The latency of model routing operations in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

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

	// PIIViolations tracks PII policy violations by model and PII data type
	PIIViolations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_pii_violations_total",
			Help: "The total number of PII policy violations by model and PII data type",
		},
		[]string{"model", "pii_type"},
	)

	// HallucinationDetectionLatency tracks the latency of hallucination detection
	HallucinationDetectionLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_hallucination_detection_latency_seconds",
			Help:    "The latency of hallucination detection operations in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// UnverifiedFactualResponses tracks responses that needed fact-checking but had no context
	UnverifiedFactualResponses = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_unverified_factual_responses_total",
			Help: "The total number of factual responses that could not be verified due to missing tool context",
		},
	)

	// ReasoningDecisions tracks the reasoning mode decision outcome by category, model, and effort
	ReasoningDecisions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_decisions_total",
			Help: "The total number of reasoning mode decisions by category, model, and effort",
		},
		[]string{"category", "model", "enabled", "effort"},
	)

	// ReasoningTemplateUsage tracks usage of model-family-specific template parameters
	ReasoningTemplateUsage = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_template_usage_total",
			Help: "The total number of times a model family template parameter was applied",
		},
		[]string{"family", "param"},
	)

	// ReasoningEffortUsage tracks the distribution of reasoning efforts by model family
	ReasoningEffortUsage = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_reasoning_effort_usage_total",
			Help: "The total number of times a reasoning effort level was set per model family",
		},
		[]string{"family", "effort"},
	)

	// EntropyClassificationDecisions tracks entropy-based reasoning decisions
	EntropyClassificationDecisions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_entropy_classification_decisions_total",
			Help: "The total number of entropy-based classification decisions by uncertainty level and reasoning outcome",
		},
		[]string{"uncertainty_level", "reasoning_enabled", "decision_reason", "top_category"},
	)

	// EntropyValues tracks the distribution of entropy values in classifications
	EntropyValues = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_entropy_values",
			Help:    "Distribution of Shannon entropy values in classification decisions",
			Buckets: []float64{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0},
		},
		[]string{"category", "classification_type"},
	)

	// ClassificationConfidence tracks confidence scores from probability distributions
	ClassificationConfidence = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_classification_confidence",
			Help:    "Distribution of classification confidence scores",
			Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"category", "classification_method"},
	)

	// EntropyClassificationLatency tracks the latency of entropy-based classification
	EntropyClassificationLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_entropy_classification_latency_seconds",
			Help:    "The latency of entropy-based classification operations in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0},
		},
	)

	// ProbabilityDistributionQuality tracks quality metrics of probability distributions
	ProbabilityDistributionQuality = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_probability_distribution_quality_total",
			Help: "Quality indicators for probability distributions from classification models",
		},
		[]string{"quality_check", "status"},
	)

	// EntropyFallbackUsage tracks when entropy-based routing falls back to traditional methods
	EntropyFallbackUsage = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_entropy_fallback_usage_total",
			Help: "The number of times entropy-based routing falls back to traditional classification",
		},
		[]string{"fallback_reason", "fallback_strategy"},
	)

	// Signal extraction metrics
	// SignalExtractionTotal tracks the total number of signal extractions by type and name
	SignalExtractionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_signal_extraction_total",
			Help: "Total number of signal extractions by type and name",
		},
		[]string{"signal_type", "signal_name"},
	)

	// SignalExtractionLatency tracks the latency of signal extraction by type
	SignalExtractionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_signal_extraction_latency_seconds",
			Help:    "Latency of signal extraction by type in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"signal_type"},
	)

	// SignalMatchTotal tracks the total number of signal matches by type and name
	SignalMatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_signal_match_total",
			Help: "Total number of signal matches by type and name",
		},
		[]string{"signal_type", "signal_name"},
	)

	// Decision evaluation metrics
	// DecisionEvaluationTotal tracks the total number of decision evaluations
	DecisionEvaluationTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_decision_evaluation_total",
			Help: "Total number of decision evaluations",
		},
	)

	// DecisionEvaluationLatency tracks the latency of decision evaluation
	DecisionEvaluationLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_decision_evaluation_latency_seconds",
			Help:    "Latency of decision evaluation in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// DecisionMatchTotal tracks the total number of decision matches by decision name
	DecisionMatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_decision_match_total",
			Help: "Total number of decision matches by decision name",
		},
		[]string{"decision_name"},
	)

	// DecisionConfidence tracks the distribution of decision confidence scores
	DecisionConfidence = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_decision_confidence",
			Help:    "Distribution of decision confidence scores by decision name",
			Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"decision_name"},
	)

	// Plugin execution metrics
	// PluginExecutionTotal tracks the total number of plugin executions by type, decision, and status
	PluginExecutionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_plugin_execution_total",
			Help: "Total number of plugin executions by type, decision, and status",
		},
		[]string{"plugin_type", "decision_name", "status"},
	)

	// PluginExecutionLatency tracks the latency of plugin execution by type
	PluginExecutionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_plugin_execution_latency_seconds",
			Help:    "Latency of plugin execution by type in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"plugin_type"},
	)

	// PluginExecutionErrors tracks the total number of plugin execution errors by type and reason
	PluginExecutionErrors = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_plugin_execution_errors_total",
			Help: "Total number of plugin execution errors by type and reason",
		},
		[]string{"plugin_type", "error_reason"},
	)

	// RAG (Retrieval-Augmented Generation) metrics
	RAGRetrievalAttempts = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_retrieval_attempts_total",
			Help: "Total number of RAG retrieval attempts",
		},
		[]string{"backend", "decision", "status"}, // status: success, error
	)

	RAGRetrievalLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_retrieval_latency_seconds",
			Help:    "RAG retrieval latency in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
		},
		[]string{"backend", "decision"},
	)

	RAGSimilarityScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "rag_similarity_score",
			Help: "Average similarity score of retrieved documents",
		},
		[]string{"backend", "decision"},
	)

	RAGContextLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_context_length_chars",
			Help:    "Length of retrieved context in characters",
			Buckets: []float64{100, 500, 1000, 2000, 5000, 10000, 20000},
		},
		[]string{"backend", "decision"},
	)

	RAGCacheHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_cache_hits_total",
			Help: "Total number of RAG cache hits",
		},
		[]string{"backend"},
	)

	RAGCacheMisses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_cache_misses_total",
			Help: "Total number of RAG cache misses",
		},
		[]string{"backend"},
	)

	// ContextTokenCount tracks the distribution of input token counts for context-based routing
	ContextTokenCount = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_token_count",
			Help:    "Distribution of input token counts for context-based routing",
			Buckets: []float64{100, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000},
		},
		[]string{"model", "context_level"},
	)
)

// RecordModelRequest increments the counter for requests to a specific model
func RecordModelRequest(model string) {
	if model == "" {
		model = consts.UnknownLabel
	}
	ModelRequests.WithLabelValues(model).Inc()
}

// RecordRequestError increments request error counters labeled by model and normalized reason
func RecordRequestError(model, reason string) {
	if model == "" {
		model = consts.UnknownLabel
	}
	if reason == "" {
		reason = consts.UnknownLabel
	}
	// Normalize a few common variants to canonical reasons
	switch reason {
	case "deadline_exceeded":
		reason = "timeout"
	case "upstream_500", "upstream_502", "upstream_503", "upstream_504":
		reason = "upstream_5xx"
	case "upstream_400", "upstream_401", "upstream_403", "upstream_404", "upstream_429":
		reason = "upstream_4xx"
	}
	RequestErrorsTotal.WithLabelValues(model, reason).Inc()
}

// RecordModelRouting records that a request was routed from one model to another
func RecordModelRouting(sourceModel, targetModel string) {
	if sourceModel != targetModel {
		ModelRoutingModifications.WithLabelValues(sourceModel, targetModel).Inc()
	}
}

// RecordModelTokens adds the number of tokens used by a specific model
func RecordModelTokens(model string, tokens float64) {
	ModelTokens.WithLabelValues(model).Add(tokens)
}

// RecordModelCost records the cost attributed to a specific model with a currency label
func RecordModelCost(model string, currency string, amount float64) {
	if amount < 0 {
		return
	}
	if currency == "" {
		currency = "USD"
	}
	ModelCost.WithLabelValues(model, currency).Add(amount)
}

// RecordRoutingReasonCode increments the counter for a routing decision reason code and model
func RecordRoutingReasonCode(reasonCode, model string) {
	if reasonCode == "" {
		reasonCode = consts.UnknownLabel
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	RoutingReasonCodes.WithLabelValues(reasonCode, model).Inc()
}

// RecordModelTokensDetailed records detailed token usage (prompt and completion)
func RecordModelTokensDetailed(model string, promptTokens, completionTokens float64) {
	// Record in both the aggregated and detailed metrics
	totalTokens := promptTokens + completionTokens
	ModelTokens.WithLabelValues(model).Add(totalTokens)
	ModelPromptTokens.WithLabelValues(model).Add(promptTokens)
	ModelCompletionTokens.WithLabelValues(model).Add(completionTokens)

	// Also record per-request histograms for visibility into distribution
	if model == "" {
		model = consts.UnknownLabel
	}
	PromptTokensPerRequest.WithLabelValues(model).Observe(promptTokens)
	CompletionTokensPerRequest.WithLabelValues(model).Observe(completionTokens)
}

// RecordModelCompletionLatency records the latency of a model completion
func RecordModelCompletionLatency(model string, seconds float64) {
	ModelCompletionLatency.WithLabelValues(model).Observe(seconds)
}

// RecordModelTTFT records time to first token for a model
func RecordModelTTFT(model string, seconds float64) {
	if seconds <= 0 {
		return
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	ModelTTFT.WithLabelValues(model).Observe(seconds)
}

// RecordModelTPOT records time per output token (seconds per token) for a model
func RecordModelTPOT(model string, secondsPerToken float64) {
	if secondsPerToken <= 0 {
		return
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	ModelTPOT.WithLabelValues(model).Observe(secondsPerToken)
}

// RecordModelRoutingLatency records the latency of model routing
func RecordModelRoutingLatency(seconds float64) {
	ModelRoutingLatency.Observe(seconds)
}

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

// RecordPIIViolation records a PII policy violation for a specific model and PII data type
func RecordPIIViolation(model string, piiType string) {
	PIIViolations.WithLabelValues(model, piiType).Inc()
}

// RecordPIIViolations records multiple PII policy violations for a specific model
func RecordPIIViolations(model string, piiTypes []string) {
	for _, piiType := range piiTypes {
		PIIViolations.WithLabelValues(model, piiType).Inc()
	}
}

// RecordHallucinationDetectionLatency records the latency for hallucination detection
func RecordHallucinationDetectionLatency(seconds float64) {
	HallucinationDetectionLatency.Observe(seconds)
}

// RecordUnverifiedFactualResponse records when a factual response could not be verified
func RecordUnverifiedFactualResponse() {
	UnverifiedFactualResponses.Inc()
}

// Batch Classification Metrics - Dynamically initialized based on configuration
var (
	BatchClassificationRequests *prometheus.CounterVec
	BatchClassificationDuration *prometheus.HistogramVec
	BatchClassificationTexts    *prometheus.CounterVec
	BatchClassificationErrors   *prometheus.CounterVec
	ConcurrentGoroutines        *prometheus.GaugeVec
	BatchSizeDistribution       *prometheus.HistogramVec
)

// Default batch size ranges - used only when configuration is missing
var DefaultBatchSizeRanges = []config.BatchSizeRangeConfig{
	{Min: 1, Max: 1, Label: "1"},
	{Min: 2, Max: 5, Label: "2-5"},
	{Min: 6, Max: 10, Label: "6-10"},
	{Min: 11, Max: 20, Label: "11-20"},
	{Min: 21, Max: 50, Label: "21-50"},
	{Min: 51, Max: -1, Label: "50+"}, // -1 means no upper limit
}

// Uses ranges from configuration file
func GetBatchSizeRange(size int) string {
	config := GetBatchMetricsConfig()
	ranges := config.BatchSizeRanges

	// Use default ranges if not configured
	if len(ranges) == 0 {
		ranges = DefaultBatchSizeRanges
	}

	// Find the appropriate range for the given size
	for _, r := range ranges {
		if size >= r.Min && (r.Max == -1 || size <= r.Max) {
			return r.Label
		}
	}

	// Fallback for unexpected cases
	return consts.UnknownLabel
}

// GetBatchSizeRangeFromBuckets generates range labels based on size buckets
func GetBatchSizeRangeFromBuckets(size int, buckets []float64) string {
	if len(buckets) == 0 {
		return GetBatchSizeRange(size) // fallback to default ranges
	}

	sizeFloat := float64(size)

	// Find which bucket this size falls into
	for i, bucket := range buckets {
		if sizeFloat <= bucket {
			if i == 0 {
				return fmt.Sprintf("≤%.0f", bucket)
			}
			prevBucket := buckets[i-1]
			if prevBucket == bucket-1 {
				return fmt.Sprintf("%.0f", bucket)
			}
			return fmt.Sprintf("%.0f-%.0f", prevBucket+1, bucket)
		}
	}

	// Size is larger than the largest bucket
	lastBucket := buckets[len(buckets)-1]
	return fmt.Sprintf("%.0f+", lastBucket)
}

// RecordBatchClassificationRequest increments the counter for batch classification requests
func RecordBatchClassificationRequest(processingType string) {
	if !shouldCollectMetric() {
		return
	}
	BatchClassificationRequests.WithLabelValues(processingType).Inc()
}

// RecordBatchClassificationDuration records the duration of batch classification processing
func RecordBatchClassificationDuration(processingType string, batchSize int, duration float64) {
	if !shouldCollectMetric() {
		return
	}

	// Use configured range labels from config.yaml
	batchSizeRange := GetBatchSizeRange(batchSize)
	BatchClassificationDuration.WithLabelValues(processingType, batchSizeRange).Observe(duration)
}

// RecordBatchClassificationTexts adds the number of texts processed in batch classification
func RecordBatchClassificationTexts(processingType string, count int) {
	if !shouldCollectMetric() {
		return
	}
	BatchClassificationTexts.WithLabelValues(processingType).Add(float64(count))
}

// RecordBatchClassificationError increments the counter for batch classification errors
func RecordBatchClassificationError(processingType, errorType string) {
	if !shouldCollectMetric() {
		return
	}
	BatchClassificationErrors.WithLabelValues(processingType, errorType).Inc()
}

// RecordBatchSizeDistribution records the distribution of batch sizes
func RecordBatchSizeDistribution(processingType string, batchSize int) {
	if !shouldCollectMetric() {
		return
	}
	BatchSizeDistribution.WithLabelValues(processingType).Observe(float64(batchSize))
}

// GenerateExponentialBuckets creates exponential histogram buckets
func GenerateExponentialBuckets(start, factor float64, count int) []float64 {
	buckets := make([]float64, count)
	buckets[0] = start
	for i := 1; i < count; i++ {
		buckets[i] = buckets[i-1] * factor
	}
	return buckets
}

// GenerateLinearBuckets creates linear histogram buckets
func GenerateLinearBuckets(start, width float64, count int) []float64 {
	buckets := make([]float64, count)
	for i := 0; i < count; i++ {
		buckets[i] = start + float64(i)*width
	}
	return buckets
}

// GetBucketsFromConfig returns buckets from configuration
func GetBucketsFromConfig(config BatchMetricsConfig) (durationBuckets, sizeBuckets []float64) {
	// Use configured buckets or fallback to defaults
	if len(config.DurationBuckets) > 0 {
		durationBuckets = config.DurationBuckets
	} else {
		durationBuckets = FallbackDurationBuckets
	}

	if len(config.SizeBuckets) > 0 {
		sizeBuckets = config.SizeBuckets
	} else {
		sizeBuckets = FallbackSizeBuckets
	}

	return durationBuckets, sizeBuckets
}

// InitializeBatchMetrics initializes batch classification metrics with custom bucket configurations
func InitializeBatchMetrics(config BatchMetricsConfig) {
	metricsInitOnce.Do(func() {
		// Get buckets from configuration
		durationBuckets, sizeBuckets := GetBucketsFromConfig(config)

		// Initialize metrics with configuration-driven buckets
		BatchClassificationRequests = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "batch_classification_requests_total",
				Help: "Total number of batch classification requests",
			},
			[]string{"processing_type"},
		)

		BatchClassificationDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "batch_classification_duration_seconds",
				Help:    "Duration of batch classification processing",
				Buckets: durationBuckets,
			},
			[]string{"processing_type", "batch_size_range"},
		)

		BatchClassificationTexts = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "batch_classification_texts_total",
				Help: "Total number of texts processed in batch classification",
			},
			[]string{"processing_type"},
		)

		BatchClassificationErrors = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "batch_classification_errors_total",
				Help: "Total number of batch classification errors",
			},
			[]string{"processing_type", "error_type"},
		)

		ConcurrentGoroutines = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "batch_classification_concurrent_goroutines",
				Help: "Number of active goroutines in concurrent batch processing",
			},
			[]string{"batch_id"},
		)

		BatchSizeDistribution = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "batch_classification_size_distribution",
				Help:    "Distribution of batch sizes",
				Buckets: sizeBuckets,
			},
			[]string{"processing_type"},
		)
	})
}

// RecordReasoningDecision records a reasoning-mode decision for a category, model and effort
func RecordReasoningDecision(category, model string, enabled bool, effort string) {
	status := "false"
	if enabled {
		status = "true"
	}
	ReasoningDecisions.WithLabelValues(category, model, status, effort).Inc()
}

// RecordReasoningTemplateUsage records usage of a model-family-specific template parameter
func RecordReasoningTemplateUsage(family, param string) {
	if family == "" {
		family = consts.UnknownLabel
	}
	if param == "" {
		param = "none"
	}
	ReasoningTemplateUsage.WithLabelValues(family, param).Inc()
}

// RecordReasoningEffortUsage records the effort usage by model family
func RecordReasoningEffortUsage(family, effort string) {
	if family == "" {
		family = consts.UnknownLabel
	}
	if effort == "" {
		effort = "unspecified"
	}
	ReasoningEffortUsage.WithLabelValues(family, effort).Inc()
}

// RecordEntropyClassificationDecision records an entropy-based classification decision
func RecordEntropyClassificationDecision(uncertaintyLevel string, reasoningEnabled bool, decisionReason string, topCategory string) {
	if uncertaintyLevel == "" {
		uncertaintyLevel = consts.UnknownLabel
	}
	if decisionReason == "" {
		decisionReason = "unspecified"
	}
	if topCategory == "" {
		topCategory = "none"
	}

	reasoningStatus := "false"
	if reasoningEnabled {
		reasoningStatus = "true"
	}

	EntropyClassificationDecisions.WithLabelValues(uncertaintyLevel, reasoningStatus, decisionReason, topCategory).Inc()
}

// RecordEntropyValue records the entropy value for a classification
func RecordEntropyValue(category string, classificationType string, entropyValue float64) {
	if category == "" {
		category = consts.UnknownLabel
	}
	if classificationType == "" {
		classificationType = "standard"
	}

	EntropyValues.WithLabelValues(category, classificationType).Observe(entropyValue)
}

// RecordClassificationConfidence records the confidence score from classification
func RecordClassificationConfidence(category string, classificationMethod string, confidence float64) {
	if category == "" {
		category = consts.UnknownLabel
	}
	if classificationMethod == "" {
		classificationMethod = "traditional"
	}

	ClassificationConfidence.WithLabelValues(category, classificationMethod).Observe(confidence)
}

// RecordEntropyClassificationLatency records the latency of entropy-based classification
func RecordEntropyClassificationLatency(seconds float64) {
	EntropyClassificationLatency.Observe(seconds)
}

// RecordProbabilityDistributionQuality records quality checks for probability distributions
func RecordProbabilityDistributionQuality(qualityCheck string, status string) {
	if qualityCheck == "" {
		qualityCheck = consts.UnknownLabel
	}
	if status == "" {
		status = consts.UnknownLabel
	}

	ProbabilityDistributionQuality.WithLabelValues(qualityCheck, status).Inc()
}

// RecordEntropyFallback records when entropy-based routing falls back to traditional methods
func RecordEntropyFallback(fallbackReason string, fallbackStrategy string) {
	if fallbackReason == "" {
		fallbackReason = consts.UnknownLabel
	}
	if fallbackStrategy == "" {
		fallbackStrategy = "unspecified"
	}

	EntropyFallbackUsage.WithLabelValues(fallbackReason, fallbackStrategy).Inc()
}

// RecordEntropyClassificationMetrics records comprehensive entropy-based classification metrics
func RecordEntropyClassificationMetrics(
	category string,
	uncertaintyLevel string,
	entropyValue float64,
	confidence float64,
	reasoningEnabled bool,
	decisionReason string,
	topCategory string,
	latencySeconds float64,
) {
	// Record the main decision
	RecordEntropyClassificationDecision(uncertaintyLevel, reasoningEnabled, decisionReason, topCategory)

	// Record entropy value
	RecordEntropyValue(category, "entropy_based", entropyValue)

	// Record confidence
	RecordClassificationConfidence(category, "entropy_based", confidence)

	// Record latency if provided
	if latencySeconds > 0 {
		RecordEntropyClassificationLatency(latencySeconds)
	}
}

// RecordSignalExtraction records a signal extraction event
func RecordSignalExtraction(signalType, signalName string, latencySeconds float64) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalExtractionTotal.WithLabelValues(signalType, signalName).Inc()
	SignalExtractionLatency.WithLabelValues(signalType).Observe(latencySeconds)
}

// RecordSignalMatch records a signal match event
func RecordSignalMatch(signalType, signalName string) {
	if signalType == "" {
		signalType = consts.UnknownLabel
	}
	if signalName == "" {
		signalName = consts.UnknownLabel
	}
	SignalMatchTotal.WithLabelValues(signalType, signalName).Inc()
}

// RecordDecisionEvaluation records a decision evaluation event
func RecordDecisionEvaluation(latencySeconds float64) {
	DecisionEvaluationTotal.Inc()
	DecisionEvaluationLatency.Observe(latencySeconds)
}

// RecordDecisionMatch records a decision match event with confidence
func RecordDecisionMatch(decisionName string, confidence float64) {
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	DecisionMatchTotal.WithLabelValues(decisionName).Inc()
	DecisionConfidence.WithLabelValues(decisionName).Observe(confidence)
}

// RecordPluginExecution records a plugin execution event
func RecordPluginExecution(pluginType, decisionName, status string, latencySeconds float64) {
	if pluginType == "" {
		pluginType = consts.UnknownLabel
	}
	if decisionName == "" {
		decisionName = consts.UnknownLabel
	}
	if status == "" {
		status = "unknown"
	}
	PluginExecutionTotal.WithLabelValues(pluginType, decisionName, status).Inc()
	PluginExecutionLatency.WithLabelValues(pluginType).Observe(latencySeconds)
}

// RecordPluginError records a plugin execution error
func RecordPluginError(pluginType, errorReason string) {
	if pluginType == "" {
		pluginType = consts.UnknownLabel
	}
	if errorReason == "" {
		errorReason = "unknown"
	}
	PluginExecutionErrors.WithLabelValues(pluginType, errorReason).Inc()
}

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
