package metrics

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestMain ensures metrics are initialized before running tests
func TestMain(m *testing.M) {
	// Initialize metrics with default configuration for testing
	config := BatchMetricsConfig{
		Enabled:                   true,
		DetailedGoroutineTracking: true,
		HighResolutionTiming:      false,
		SampleRate:                1.0,
		DurationBuckets:           FallbackDurationBuckets,
		SizeBuckets:               FallbackSizeBuckets,
		BatchSizeRanges: []config.BatchSizeRangeConfig{
			{Min: 1, Max: 1, Label: "1"},
			{Min: 2, Max: 5, Label: "2-5"},
			{Min: 6, Max: 10, Label: "6-10"},
			{Min: 11, Max: 20, Label: "11-20"},
			{Min: 21, Max: 50, Label: "21-50"},
			{Min: 51, Max: -1, Label: "50+"},
		},
	}

	// Initialize batch metrics
	InitializeBatchMetrics(config)
	SetBatchMetricsConfig(config)

	// Run tests
	m.Run()
}

// TestBatchClassificationMetrics tests the batch classification metrics recording
func TestBatchClassificationMetrics(t *testing.T) {
	tests := []struct {
		name           string
		processingType string
		batchSize      int
		duration       float64
		errorType      string
		expectError    bool
	}{
		{
			name:           "Sequential processing metrics",
			processingType: "sequential",
			batchSize:      3,
			duration:       0.5,
			errorType:      "",
			expectError:    false,
		},
		{
			name:           "Concurrent processing metrics",
			processingType: "concurrent",
			batchSize:      10,
			duration:       1.2,
			errorType:      "",
			expectError:    false,
		},
		{
			name:           "Error case metrics",
			processingType: "sequential",
			batchSize:      5,
			duration:       0.3,
			errorType:      "classification_failed",
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(_ *testing.T) {
			// Record metrics
			RecordBatchClassificationRequest(tt.processingType)
			RecordBatchSizeDistribution(tt.processingType, tt.batchSize)
			RecordBatchClassificationDuration(tt.processingType, tt.batchSize, tt.duration)
			RecordBatchClassificationTexts(tt.processingType, tt.batchSize)

			if tt.expectError {
				RecordBatchClassificationError(tt.processingType, tt.errorType)
			}
		})
	}
}

// TestGetBatchSizeRange tests the batch size range helper function
func TestGetBatchSizeRange(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected string
	}{
		{"Single text", 1, "1"},
		{"Small batch", 3, "2-5"},
		{"Medium batch", 8, "6-10"},
		{"Large batch", 15, "11-20"},
		{"Very large batch", 35, "21-50"},
		{"Maximum batch", 100, "50+"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetBatchSizeRange(tt.size)
			if result != tt.expected {
				t.Errorf("GetBatchSizeRange(%d) = %s, want %s", tt.size, result, tt.expected)
			}
		})
	}
}

// TestConcurrentGoroutineTracking tests goroutine tracking functionality
func TestConcurrentGoroutineTracking(_ *testing.T) {
	batchID := "test_batch_123"

	// Simulate goroutine start
	ConcurrentGoroutines.WithLabelValues(batchID).Inc()

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	// Simulate goroutine end
	ConcurrentGoroutines.WithLabelValues(batchID).Dec()
}

// BenchmarkBatchClassificationMetrics benchmarks the performance impact of metrics recording
func BenchmarkBatchClassificationMetrics(b *testing.B) {
	processingType := "concurrent"
	batchSize := 10
	duration := 0.5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RecordBatchClassificationRequest(processingType)
		RecordBatchSizeDistribution(processingType, batchSize)
		RecordBatchClassificationDuration(processingType, batchSize, duration)
		RecordBatchClassificationTexts(processingType, batchSize)
	}
}

// TestMetricsIntegration tests the integration of all batch classification metrics
func TestMetricsIntegration(_ *testing.T) {
	// Simulate a complete batch processing scenario
	processingType := "concurrent"
	batchSize := 8
	batchID := "integration_test_batch"

	// Start of batch processing
	RecordBatchClassificationRequest(processingType)
	RecordBatchSizeDistribution(processingType, batchSize)

	// Simulate concurrent goroutines
	for i := 0; i < batchSize; i++ {
		ConcurrentGoroutines.WithLabelValues(batchID).Inc()
	}

	// Simulate processing completion
	duration := 1.5
	RecordBatchClassificationDuration(processingType, batchSize, duration)
	RecordBatchClassificationTexts(processingType, batchSize)

	// Simulate goroutines completion
	for i := 0; i < batchSize; i++ {
		ConcurrentGoroutines.WithLabelValues(batchID).Dec()
	}
}

// TestDefaultBatchSizeRanges tests that default batch size ranges work when configuration is empty
func TestDefaultBatchSizeRanges(t *testing.T) {
	// Save current config
	originalConfig := GetBatchMetricsConfig()

	// Set config with empty BatchSizeRanges to test defaults
	emptyConfig := BatchMetricsConfig{
		Enabled:                   true,
		DetailedGoroutineTracking: true,
		HighResolutionTiming:      false,
		SampleRate:                1.0,
		DurationBuckets:           FallbackDurationBuckets,
		SizeBuckets:               FallbackSizeBuckets,
		BatchSizeRanges:           []config.BatchSizeRangeConfig{}, // Empty - should use defaults
	}
	SetBatchMetricsConfig(emptyConfig)

	// Test that default ranges are used
	tests := []struct {
		name     string
		size     int
		expected string
	}{
		{"Single text (default)", 1, "1"},
		{"Small batch (default)", 3, "2-5"},
		{"Medium batch (default)", 8, "6-10"},
		{"Large batch (default)", 15, "11-20"},
		{"Very large batch (default)", 35, "21-50"},
		{"Maximum batch (default)", 100, "50+"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetBatchSizeRange(tt.size)
			if result != tt.expected {
				t.Errorf("GetBatchSizeRange(%d) with empty config = %s, want %s", tt.size, result, tt.expected)
			}
		})
	}

	// Restore original config
	SetBatchMetricsConfig(originalConfig)
}

// TestCustomBatchSizeRanges tests that custom batch size ranges override defaults
func TestCustomBatchSizeRanges(t *testing.T) {
	// Save current config
	originalConfig := GetBatchMetricsConfig()

	// Set config with custom BatchSizeRanges
	customConfig := BatchMetricsConfig{
		Enabled:                   true,
		DetailedGoroutineTracking: true,
		HighResolutionTiming:      false,
		SampleRate:                1.0,
		DurationBuckets:           FallbackDurationBuckets,
		SizeBuckets:               FallbackSizeBuckets,
		BatchSizeRanges: []config.BatchSizeRangeConfig{
			{Min: 1, Max: 10, Label: "small"},
			{Min: 11, Max: 100, Label: "medium"},
			{Min: 101, Max: -1, Label: "large"},
		},
	}
	SetBatchMetricsConfig(customConfig)

	// Test that custom ranges are used
	tests := []struct {
		name     string
		size     int
		expected string
	}{
		{"Custom small range", 5, "small"},
		{"Custom medium range", 50, "medium"},
		{"Custom large range", 200, "large"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetBatchSizeRange(tt.size)
			if result != tt.expected {
				t.Errorf("GetBatchSizeRange(%d) with custom config = %s, want %s", tt.size, result, tt.expected)
			}
		})
	}

	// Restore original config
	SetBatchMetricsConfig(originalConfig)
}

// TestUnverifiedFactualResponseMetric tests the unverified factual response counter
func TestUnverifiedFactualResponseMetric(_ *testing.T) {
	// Record multiple unverified responses
	for i := 0; i < 5; i++ {
		RecordUnverifiedFactualResponse()
	}
	// The test passes if no panic occurs - Prometheus counters are monotonic
}
