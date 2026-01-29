package scoring

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// WindowedMetricsProvider implements MetricsProvider using the windowed metrics system
type WindowedMetricsProvider struct {
	// costTracker tracks per-model cost data
	costTracker map[string]*CostData
	costMutex   sync.RWMutex

	// requestTracker tracks per-model request data for scoring
	requestTracker map[string]*RequestTracker
	trackerMutex   sync.RWMutex

	// maxModels limits tracked models to prevent memory issues
	maxModels int
}

// CostData tracks cost information for a model
type CostData struct {
	TotalCost   float64
	TotalTokens int64
	LastUpdated time.Time
}

// RequestTracker tracks request metrics for scoring within a time window
type RequestTracker struct {
	// Ring buffer for request data
	requests []RequestRecord
	head     int
	size     int
	capacity int
	mutex    sync.RWMutex
}

// RequestRecord represents a single request's metrics
type RequestRecord struct {
	Timestamp        time.Time
	LatencySeconds   float64
	PromptTokens     int64
	CompletionTokens int64
	IsError          bool
	Cost             float64
}

// NewWindowedMetricsProvider creates a new provider
func NewWindowedMetricsProvider(maxModels int) *WindowedMetricsProvider {
	if maxModels <= 0 {
		maxModels = 100
	}
	return &WindowedMetricsProvider{
		costTracker:    make(map[string]*CostData),
		requestTracker: make(map[string]*RequestTracker),
		maxModels:      maxModels,
	}
}

// RecordRequest records a request for metrics tracking
func (p *WindowedMetricsProvider) RecordRequest(model string, latencySeconds float64, promptTokens, completionTokens int64, isError bool, cost float64) {
	p.trackerMutex.Lock()
	tracker, exists := p.requestTracker[model]
	if !exists {
		if len(p.requestTracker) >= p.maxModels {
			p.trackerMutex.Unlock()
			return
		}
		tracker = newRequestTracker(10000) // 10k capacity per model
		p.requestTracker[model] = tracker
	}
	p.trackerMutex.Unlock()

	record := RequestRecord{
		Timestamp:        time.Now(),
		LatencySeconds:   latencySeconds,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		IsError:          isError,
		Cost:             cost,
	}
	tracker.add(record)

	// Update cost tracking
	p.costMutex.Lock()
	if cd, ok := p.costTracker[model]; ok {
		cd.TotalCost += cost
		cd.TotalTokens += promptTokens + completionTokens
		cd.LastUpdated = time.Now()
	} else if len(p.costTracker) < p.maxModels {
		p.costTracker[model] = &CostData{
			TotalCost:   cost,
			TotalTokens: promptTokens + completionTokens,
			LastUpdated: time.Now(),
		}
	}
	p.costMutex.Unlock()
}

// GetModelMetrics returns metrics for a specific model over the given time window
func (p *WindowedMetricsProvider) GetModelMetrics(model string, window time.Duration) (ModelMetrics, error) {
	p.trackerMutex.RLock()
	tracker, exists := p.requestTracker[model]
	p.trackerMutex.RUnlock()

	if !exists {
		return ModelMetrics{}, nil
	}

	since := time.Now().Add(-window)
	records := tracker.getRecordsSince(since)

	return p.computeMetrics(model, records), nil
}

// GetAllModelMetrics returns metrics for all tracked models
func (p *WindowedMetricsProvider) GetAllModelMetrics(window time.Duration) (map[string]ModelMetrics, error) {
	p.trackerMutex.RLock()
	models := make([]string, 0, len(p.requestTracker))
	for model := range p.requestTracker {
		models = append(models, model)
	}
	p.trackerMutex.RUnlock()

	result := make(map[string]ModelMetrics, len(models))
	since := time.Now().Add(-window)

	for _, model := range models {
		p.trackerMutex.RLock()
		tracker := p.requestTracker[model]
		p.trackerMutex.RUnlock()

		if tracker != nil {
			records := tracker.getRecordsSince(since)
			if len(records) > 0 {
				result[model] = p.computeMetrics(model, records)
			}
		}
	}

	return result, nil
}

// computeMetrics computes ModelMetrics from request records
func (p *WindowedMetricsProvider) computeMetrics(model string, records []RequestRecord) ModelMetrics {
	if len(records) == 0 {
		return ModelMetrics{}
	}

	var totalLatency float64
	var totalTokens int64
	var totalCost float64
	var errorCount int

	latencies := make([]float64, 0, len(records))

	for _, r := range records {
		totalLatency += r.LatencySeconds
		totalTokens += r.PromptTokens + r.CompletionTokens
		totalCost += r.Cost
		latencies = append(latencies, r.LatencySeconds)
		if r.IsError {
			errorCount++
		}
	}

	requestCount := len(records)
	avgLatency := totalLatency / float64(requestCount)
	errorRate := float64(errorCount) / float64(requestCount)

	// Compute P95 latency
	p95Latency := computePercentile(latencies, 0.95)

	// Compute cost per 1K tokens
	var costPerKTokens float64
	if totalTokens > 0 {
		costPerKTokens = (totalCost / float64(totalTokens)) * 1000
	}

	return ModelMetrics{
		RequestCount:      requestCount,
		ErrorCount:        errorCount,
		ErrorRate:         errorRate,
		AvgLatencySeconds: avgLatency,
		P95LatencySeconds: p95Latency,
		TotalTokens:       totalTokens,
		TotalCost:         totalCost,
		CostPerKTokens:    costPerKTokens,
	}
}

// Helper functions for RequestTracker

func newRequestTracker(capacity int) *RequestTracker {
	return &RequestTracker{
		requests: make([]RequestRecord, capacity),
		capacity: capacity,
	}
}

func (rt *RequestTracker) add(record RequestRecord) {
	rt.mutex.Lock()
	defer rt.mutex.Unlock()

	rt.requests[rt.head] = record
	rt.head = (rt.head + 1) % rt.capacity
	if rt.size < rt.capacity {
		rt.size++
	}
}

func (rt *RequestTracker) getRecordsSince(since time.Time) []RequestRecord {
	rt.mutex.RLock()
	defer rt.mutex.RUnlock()

	result := make([]RequestRecord, 0, rt.size)
	for i := 0; i < rt.size; i++ {
		idx := (rt.head - rt.size + i + rt.capacity) % rt.capacity
		if !rt.requests[idx].Timestamp.Before(since) {
			result = append(result, rt.requests[idx])
		}
	}
	return result
}

// computePercentile computes the given percentile from a slice of values
func computePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Copy and sort
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sortFloat64s(sorted)

	// Calculate index
	index := percentile * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1

	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}

	// Linear interpolation
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// sortFloat64s sorts a slice of float64 in ascending order
func sortFloat64s(a []float64) {
	if len(a) < 12 {
		// Insertion sort for small slices
		for i := 1; i < len(a); i++ {
			for j := i; j > 0 && a[j] < a[j-1]; j-- {
				a[j], a[j-1] = a[j-1], a[j]
			}
		}
		return
	}
	quickSort(a, 0, len(a)-1)
}

func quickSort(a []float64, low, high int) {
	if low < high {
		p := partition(a, low, high)
		quickSort(a, low, p-1)
		quickSort(a, p+1, high)
	}
}

func partition(a []float64, low, high int) int {
	pivot := a[high]
	i := low - 1
	for j := low; j < high; j++ {
		if a[j] <= pivot {
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[high] = a[high], a[i+1]
	return i + 1
}

// Global metrics provider instance
var (
	globalMetricsProvider      *WindowedMetricsProvider
	globalMetricsProviderMutex sync.RWMutex
)

// InitializeMetricsProvider initializes the global metrics provider
func InitializeMetricsProvider(maxModels int) {
	globalMetricsProviderMutex.Lock()
	defer globalMetricsProviderMutex.Unlock()
	globalMetricsProvider = NewWindowedMetricsProvider(maxModels)
	logging.Infof("Initialized scoring metrics provider with maxModels=%d", maxModels)
}

// GetMetricsProvider returns the global metrics provider
func GetMetricsProvider() *WindowedMetricsProvider {
	globalMetricsProviderMutex.RLock()
	defer globalMetricsProviderMutex.RUnlock()
	return globalMetricsProvider
}

// RecordModelRequest is a convenience function to record a request to the global provider
func RecordModelRequest(model string, latencySeconds float64, promptTokens, completionTokens int64, isError bool, cost float64) {
	globalMetricsProviderMutex.RLock()
	provider := globalMetricsProvider
	globalMetricsProviderMutex.RUnlock()

	if provider != nil {
		provider.RecordRequest(model, latencySeconds, promptTokens, completionTokens, isError, cost)
	}
}
