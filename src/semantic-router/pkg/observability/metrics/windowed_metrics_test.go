package metrics

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestNewWindowedMetricsManager tests the creation of a new WindowedMetricsManager
func TestNewWindowedMetricsManager(t *testing.T) {
	tests := []struct {
		name            string
		config          config.WindowedMetricsConfig
		wantTimeWindows int
		wantInterval    time.Duration
		wantMaxModels   int
	}{
		{
			name: "Default configuration",
			config: config.WindowedMetricsConfig{
				Enabled: true,
			},
			wantTimeWindows: 5, // Default: 1m, 5m, 15m, 1h, 24h
			wantInterval:    DefaultUpdateInterval,
			wantMaxModels:   DefaultMaxModels,
		},
		{
			name: "Custom time windows",
			config: config.WindowedMetricsConfig{
				Enabled:     true,
				TimeWindows: []string{"30s", "2m", "10m"},
			},
			wantTimeWindows: 3,
			wantInterval:    DefaultUpdateInterval,
			wantMaxModels:   DefaultMaxModels,
		},
		{
			name: "Custom update interval",
			config: config.WindowedMetricsConfig{
				Enabled:        true,
				UpdateInterval: "5s",
			},
			wantTimeWindows: 5,
			wantInterval:    5 * time.Second,
			wantMaxModels:   DefaultMaxModels,
		},
		{
			name: "Custom max models",
			config: config.WindowedMetricsConfig{
				Enabled:   true,
				MaxModels: 50,
			},
			wantTimeWindows: 5,
			wantInterval:    DefaultUpdateInterval,
			wantMaxModels:   50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager, err := NewWindowedMetricsManager(tt.config)
			if err != nil {
				t.Fatalf("NewWindowedMetricsManager() error = %v", err)
			}

			if len(manager.timeWindows) != tt.wantTimeWindows {
				t.Errorf("timeWindows count = %d, want %d", len(manager.timeWindows), tt.wantTimeWindows)
			}

			if manager.updateInterval != tt.wantInterval {
				t.Errorf("updateInterval = %v, want %v", manager.updateInterval, tt.wantInterval)
			}

			if manager.maxModels != tt.wantMaxModels {
				t.Errorf("maxModels = %d, want %d", manager.maxModels, tt.wantMaxModels)
			}
		})
	}
}

// TestRequestRingBuffer tests the ring buffer functionality
func TestRequestRingBuffer(t *testing.T) {
	rb := NewRequestRingBuffer(5)

	// Add 3 items
	now := time.Now()
	for i := 0; i < 3; i++ {
		rb.Add(RequestData{
			Timestamp:      now.Add(time.Duration(i) * time.Second),
			Model:          "model1",
			LatencySeconds: float64(i),
		})
	}

	// Should have 3 items
	data := rb.GetDataSince(now.Add(-time.Hour))
	if len(data) != 3 {
		t.Errorf("GetDataSince() count = %d, want 3", len(data))
	}

	// Add 5 more items (should wrap around)
	for i := 0; i < 5; i++ {
		rb.Add(RequestData{
			Timestamp:      now.Add(time.Duration(10+i) * time.Second),
			Model:          "model2",
			LatencySeconds: float64(10 + i),
		})
	}

	// Should have 5 items (capacity limit)
	data = rb.GetDataSince(now.Add(-time.Hour))
	if len(data) != 5 {
		t.Errorf("GetDataSince() count after wrap = %d, want 5", len(data))
	}

	// Verify data is from model2 (most recent)
	for _, d := range data {
		if d.Model != "model2" {
			t.Errorf("Expected model2, got %s", d.Model)
		}
	}
}

// TestRequestRingBufferTimeBased tests time-based filtering
func TestRequestRingBufferTimeBased(t *testing.T) {
	rb := NewRequestRingBuffer(100)

	now := time.Now()

	// Add items across different time ranges
	// Old items (2 hours ago)
	for i := 0; i < 10; i++ {
		rb.Add(RequestData{
			Timestamp:      now.Add(-2 * time.Hour),
			Model:          "model1",
			LatencySeconds: 1.0,
		})
	}

	// Recent items (5 minutes ago)
	for i := 0; i < 5; i++ {
		rb.Add(RequestData{
			Timestamp:      now.Add(-5 * time.Minute),
			Model:          "model1",
			LatencySeconds: 2.0,
		})
	}

	// Very recent items (30 seconds ago)
	for i := 0; i < 3; i++ {
		rb.Add(RequestData{
			Timestamp:      now.Add(-30 * time.Second),
			Model:          "model1",
			LatencySeconds: 3.0,
		})
	}

	// Query for last minute
	data := rb.GetDataSince(now.Add(-1 * time.Minute))
	if len(data) != 3 {
		t.Errorf("GetDataSince(1 minute) count = %d, want 3", len(data))
	}

	// Query for last 15 minutes
	data = rb.GetDataSince(now.Add(-15 * time.Minute))
	if len(data) != 8 { // 5 + 3
		t.Errorf("GetDataSince(15 minutes) count = %d, want 8", len(data))
	}

	// Query for last 24 hours
	data = rb.GetDataSince(now.Add(-24 * time.Hour))
	if len(data) != 18 { // 10 + 5 + 3
		t.Errorf("GetDataSince(24 hours) count = %d, want 18", len(data))
	}
}

// TestComputePercentile tests percentile calculation
func TestComputePercentile(t *testing.T) {
	tests := []struct {
		name       string
		values     []float64
		percentile float64
		want       float64
		tolerance  float64
	}{
		{
			name:       "Empty values",
			values:     []float64{},
			percentile: 0.5,
			want:       0,
			tolerance:  0.001,
		},
		{
			name:       "Single value",
			values:     []float64{5.0},
			percentile: 0.5,
			want:       5.0,
			tolerance:  0.001,
		},
		{
			name:       "P50 of sorted sequence",
			values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			percentile: 0.5,
			want:       5.5, // Interpolated median
			tolerance:  0.1,
		},
		{
			name:       "P95 of sorted sequence",
			values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			percentile: 0.95,
			want:       9.55,
			tolerance:  0.1,
		},
		{
			name:       "P99 of sorted sequence",
			values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			percentile: 0.99,
			want:       9.91,
			tolerance:  0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computePercentile(tt.values, tt.percentile)
			diff := got - tt.want
			if diff < 0 {
				diff = -diff
			}
			if diff > tt.tolerance {
				t.Errorf("computePercentile() = %v, want %v (tolerance %v)", got, tt.want, tt.tolerance)
			}
		})
	}
}

// TestSortFloat64s tests the sorting function
func TestSortFloat64s(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []float64
	}{
		{
			name:  "Empty slice",
			input: []float64{},
			want:  []float64{},
		},
		{
			name:  "Single element",
			input: []float64{5.0},
			want:  []float64{5.0},
		},
		{
			name:  "Already sorted",
			input: []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			want:  []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		},
		{
			name:  "Reverse sorted",
			input: []float64{5.0, 4.0, 3.0, 2.0, 1.0},
			want:  []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		},
		{
			name:  "Random order",
			input: []float64{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0},
			want:  []float64{1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0},
		},
		{
			name:  "Large slice (tests quicksort path)",
			input: []float64{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			want:  []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]float64, len(tt.input))
			copy(input, tt.input)
			sortFloat64s(input)

			if len(input) != len(tt.want) {
				t.Errorf("sortFloat64s() length = %d, want %d", len(input), len(tt.want))
				return
			}

			for i := range input {
				if input[i] != tt.want[i] {
					t.Errorf("sortFloat64s()[%d] = %v, want %v", i, input[i], tt.want[i])
				}
			}
		})
	}
}

// TestWindowedMetricsManagerRecordRequest tests request recording
func TestWindowedMetricsManagerRecordRequest(t *testing.T) {
	manager, err := NewWindowedMetricsManager(config.WindowedMetricsConfig{
		Enabled:   true,
		MaxModels: 3,
	})
	if err != nil {
		t.Fatalf("NewWindowedMetricsManager() error = %v", err)
	}

	// Record requests for multiple models
	now := time.Now()
	for i := 0; i < 5; i++ {
		manager.RecordRequest(RequestData{
			Timestamp:        now,
			Model:            "model1",
			LatencySeconds:   0.1,
			PromptTokens:     100,
			CompletionTokens: 50,
		})
	}

	manager.RecordRequest(RequestData{
		Timestamp:        now,
		Model:            "model2",
		LatencySeconds:   0.2,
		PromptTokens:     200,
		CompletionTokens: 100,
	})

	manager.RecordRequest(RequestData{
		Timestamp:        now,
		Model:            "model3",
		LatencySeconds:   0.3,
		PromptTokens:     300,
		CompletionTokens: 150,
	})

	// This should be ignored (max models reached)
	manager.RecordRequest(RequestData{
		Timestamp:        now,
		Model:            "model4",
		LatencySeconds:   0.4,
		PromptTokens:     400,
		CompletionTokens: 200,
	})

	// Check buffer count
	manager.bufferMutex.RLock()
	bufferCount := len(manager.requestBuffers)
	manager.bufferMutex.RUnlock()

	if bufferCount != 3 {
		t.Errorf("Buffer count = %d, want 3 (max models)", bufferCount)
	}
}

// TestActiveRequestTracking tests queue depth tracking
func TestActiveRequestTracking(t *testing.T) {
	manager, err := NewWindowedMetricsManager(config.WindowedMetricsConfig{
		Enabled:              true,
		QueueDepthEstimation: true,
	})
	if err != nil {
		t.Fatalf("NewWindowedMetricsManager() error = %v", err)
	}

	// Increment active requests
	manager.IncrementActiveRequests("model1")
	manager.IncrementActiveRequests("model1")
	manager.IncrementActiveRequests("model1")

	// Check count
	manager.activeMutex.RLock()
	count := manager.activeRequests["model1"]
	manager.activeMutex.RUnlock()

	if count != 3 {
		t.Errorf("Active requests count = %d, want 3", count)
	}

	// Decrement
	manager.DecrementActiveRequests("model1")
	manager.DecrementActiveRequests("model1")

	manager.activeMutex.RLock()
	count = manager.activeRequests["model1"]
	manager.activeMutex.RUnlock()

	if count != 1 {
		t.Errorf("Active requests count after decrement = %d, want 1", count)
	}

	// Decrement beyond zero (should not go negative)
	manager.DecrementActiveRequests("model1")
	manager.DecrementActiveRequests("model1")

	manager.activeMutex.RLock()
	count = manager.activeRequests["model1"]
	manager.activeMutex.RUnlock()

	if count != 0 {
		t.Errorf("Active requests count should not go negative, got %d", count)
	}
}

// TestDisabledManager tests that disabled manager doesn't record
func TestDisabledManager(t *testing.T) {
	manager, err := NewWindowedMetricsManager(config.WindowedMetricsConfig{
		Enabled: false,
	})
	if err != nil {
		t.Fatalf("NewWindowedMetricsManager() error = %v", err)
	}

	// Record request (should be ignored)
	manager.RecordRequest(RequestData{
		Timestamp:      time.Now(),
		Model:          "model1",
		LatencySeconds: 0.1,
	})

	// Check buffer count (should be 0)
	manager.bufferMutex.RLock()
	bufferCount := len(manager.requestBuffers)
	manager.bufferMutex.RUnlock()

	if bufferCount != 0 {
		t.Errorf("Buffer count = %d, want 0 (disabled manager)", bufferCount)
	}
}

// BenchmarkRecordRequest benchmarks request recording
func BenchmarkRecordRequest(b *testing.B) {
	manager, _ := NewWindowedMetricsManager(config.WindowedMetricsConfig{
		Enabled: true,
	})

	req := RequestData{
		Timestamp:        time.Now(),
		Model:            "model1",
		LatencySeconds:   0.1,
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		manager.RecordRequest(req)
	}
}

// BenchmarkComputePercentile benchmarks percentile computation
func BenchmarkComputePercentile(b *testing.B) {
	values := make([]float64, 1000)
	for i := range values {
		values[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		computePercentile(values, 0.95)
	}
}
