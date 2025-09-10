package metrics

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

	// ModelRoutingLatency tracks the latency of model routing
	ModelRoutingLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_model_routing_latency_seconds",
			Help:    "The latency of model routing operations in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// ClassifierLatency tracks the latency of classifier invocations by type
	ClassifierLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_classifier_latency_seconds",
			Help:    "The latency of classifier invocations by type",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"classifier"},
	)

	// CacheHits tracks cache hits and misses
	CacheHits = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_cache_hits_total",
			Help: "The total number of cache hits",
		},
	)

	// CategoryClassifications tracks the number of times each category is classified
	CategoryClassifications = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_category_classifications_total",
			Help: "The total number of times each category is classified",
		},
		[]string{"category"},
	)

	// PIIViolations tracks PII policy violations by model and PII data type
	PIIViolations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_pii_violations_total",
			Help: "The total number of PII policy violations by model and PII data type",
		},
		[]string{"model", "pii_type"},
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
)

// RecordModelRequest increments the counter for requests to a specific model
func RecordModelRequest(model string) {
	ModelRequests.WithLabelValues(model).Inc()
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
		reasonCode = "unknown"
	}
	if model == "" {
		model = "unknown"
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
}

// RecordModelCompletionLatency records the latency of a model completion
func RecordModelCompletionLatency(model string, seconds float64) {
	ModelCompletionLatency.WithLabelValues(model).Observe(seconds)
}

// RecordModelRoutingLatency records the latency of model routing
func RecordModelRoutingLatency(seconds float64) {
	ModelRoutingLatency.Observe(seconds)
}

// RecordCacheHit records a cache hit
func RecordCacheHit() {
	CacheHits.Inc()
}

// RecordCategoryClassification increments the gauge for a specific category classification
func RecordCategoryClassification(category string) {
	CategoryClassifications.WithLabelValues(category).Inc()
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

// RecordClassifierLatency records the latency for a classifier invocation by type
func RecordClassifierLatency(classifier string, seconds float64) {
	ClassifierLatency.WithLabelValues(classifier).Observe(seconds)
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
	return "unknown"
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
		family = "unknown"
	}
	if param == "" {
		param = "none"
	}
	ReasoningTemplateUsage.WithLabelValues(family, param).Inc()
}

// RecordReasoningEffortUsage records the effort usage by model family
func RecordReasoningEffortUsage(family, effort string) {
	if family == "" {
		family = "unknown"
	}
	if effort == "" {
		effort = "unspecified"
	}
	ReasoningEffortUsage.WithLabelValues(family, effort).Inc()
}
