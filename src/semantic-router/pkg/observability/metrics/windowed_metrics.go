package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// Default time windows for windowed metrics
var DefaultTimeWindows = []string{"1m", "5m", "15m", "1h", "24h"}

// Default update interval for computing windowed metrics
const DefaultUpdateInterval = 10 * time.Second

// Default maximum models to track
const DefaultMaxModels = 100

// WindowedMetricsManager manages time-windowed metrics for load balancing
type WindowedMetricsManager struct {
	config         config.WindowedMetricsConfig
	timeWindows    []time.Duration
	windowLabels   []string
	updateInterval time.Duration
	maxModels      int

	// Ring buffers for storing request data per model
	requestBuffers map[string]*RequestRingBuffer
	bufferMutex    sync.RWMutex

	// Active request tracking for queue depth estimation
	activeRequests map[string]int64
	activeMutex    sync.RWMutex

	// Stop channel for background goroutine
	stopChan chan struct{}
	running  bool
}

// RequestData represents a single request's data for windowed tracking
type RequestData struct {
	Timestamp        time.Time
	Model            string
	LatencySeconds   float64
	PromptTokens     int64
	CompletionTokens int64
	IsError          bool
	IsTimeout        bool
}

// RequestRingBuffer is a time-based ring buffer for storing request data
type RequestRingBuffer struct {
	data     []RequestData
	head     int
	size     int
	capacity int
	mutex    sync.RWMutex
}

// NewRequestRingBuffer creates a new ring buffer with the given capacity
func NewRequestRingBuffer(capacity int) *RequestRingBuffer {
	return &RequestRingBuffer{
		data:     make([]RequestData, capacity),
		capacity: capacity,
	}
}

// Add adds a request to the ring buffer
func (rb *RequestRingBuffer) Add(req RequestData) {
	rb.mutex.Lock()
	defer rb.mutex.Unlock()

	rb.data[rb.head] = req
	rb.head = (rb.head + 1) % rb.capacity
	if rb.size < rb.capacity {
		rb.size++
	}
}

// GetDataSince returns all request data since the given time
func (rb *RequestRingBuffer) GetDataSince(since time.Time) []RequestData {
	rb.mutex.RLock()
	defer rb.mutex.RUnlock()

	result := make([]RequestData, 0, rb.size)
	for i := 0; i < rb.size; i++ {
		idx := (rb.head - rb.size + i + rb.capacity) % rb.capacity
		if !rb.data[idx].Timestamp.Before(since) {
			result = append(result, rb.data[idx])
		}
	}
	return result
}

// Prometheus metrics for windowed model tracking
var (
	// ModelLatencyWindowed tracks latency by model and time window
	ModelLatencyWindowed *prometheus.GaugeVec

	// ModelRequestsWindowed tracks request counts by model and time window
	ModelRequestsWindowed *prometheus.GaugeVec

	// ModelTokensWindowed tracks token throughput by model, token type, and time window
	ModelTokensWindowed *prometheus.GaugeVec

	// ModelUtilization tracks utilization percentage by model and time window
	ModelUtilization *prometheus.GaugeVec

	// ModelQueueDepth tracks estimated queue depth by model
	ModelQueueDepth *prometheus.GaugeVec

	// ModelErrorRate tracks error rate by model and time window
	ModelErrorRate *prometheus.GaugeVec

	// ModelLatencyP50 tracks P50 latency by model and time window
	ModelLatencyP50 *prometheus.GaugeVec

	// ModelLatencyP95 tracks P95 latency by model and time window
	ModelLatencyP95 *prometheus.GaugeVec

	// ModelLatencyP99 tracks P99 latency by model and time window
	ModelLatencyP99 *prometheus.GaugeVec

	windowedMetricsInitOnce sync.Once
)

// Global instance of WindowedMetricsManager
var (
	globalWindowedManager      *WindowedMetricsManager
	globalWindowedManagerMutex sync.RWMutex
)

// InitializeWindowedMetrics initializes the windowed metrics system
func InitializeWindowedMetrics(cfg config.WindowedMetricsConfig) error {
	windowedMetricsInitOnce.Do(func() {
		// Initialize Prometheus metrics
		ModelLatencyWindowed = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_latency_windowed_seconds",
				Help: "Average latency by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelRequestsWindowed = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_requests_windowed_total",
				Help: "Total requests by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelTokensWindowed = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_tokens_windowed_total",
				Help: "Total tokens by model, token type, and time window",
			},
			[]string{"model", "token_type", "time_window"},
		)

		ModelUtilization = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_utilization_percentage",
				Help: "Estimated utilization percentage by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelQueueDepth = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_queue_depth_estimated",
				Help: "Estimated queue depth by model",
			},
			[]string{"model"},
		)

		ModelErrorRate = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_error_rate_windowed",
				Help: "Error rate by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelLatencyP50 = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_latency_p50_windowed_seconds",
				Help: "P50 latency by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelLatencyP95 = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_latency_p95_windowed_seconds",
				Help: "P95 latency by model and time window",
			},
			[]string{"model", "time_window"},
		)

		ModelLatencyP99 = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_latency_p99_windowed_seconds",
				Help: "P99 latency by model and time window",
			},
			[]string{"model", "time_window"},
		)
	})

	// Create and start the manager
	manager, err := NewWindowedMetricsManager(cfg)
	if err != nil {
		return err
	}

	globalWindowedManagerMutex.Lock()
	globalWindowedManager = manager
	globalWindowedManagerMutex.Unlock()

	manager.Start()
	return nil
}

// NewWindowedMetricsManager creates a new WindowedMetricsManager
func NewWindowedMetricsManager(cfg config.WindowedMetricsConfig) (*WindowedMetricsManager, error) {
	// Parse time windows
	windowStrings := cfg.TimeWindows
	if len(windowStrings) == 0 {
		windowStrings = DefaultTimeWindows
	}

	timeWindows := make([]time.Duration, 0, len(windowStrings))
	windowLabels := make([]string, 0, len(windowStrings))
	for _, ws := range windowStrings {
		d, parseErr := time.ParseDuration(ws)
		if parseErr != nil {
			// Skip invalid durations
			continue
		}
		timeWindows = append(timeWindows, d)
		windowLabels = append(windowLabels, ws)
	}

	// Parse update interval
	updateInterval := DefaultUpdateInterval
	if cfg.UpdateInterval != "" {
		if d, parseErr := time.ParseDuration(cfg.UpdateInterval); parseErr == nil {
			updateInterval = d
		}
	}

	// Set max models
	maxModels := cfg.MaxModels
	if maxModels <= 0 {
		maxModels = DefaultMaxModels
	}

	return &WindowedMetricsManager{
		config:         cfg,
		timeWindows:    timeWindows,
		windowLabels:   windowLabels,
		updateInterval: updateInterval,
		maxModels:      maxModels,
		requestBuffers: make(map[string]*RequestRingBuffer),
		activeRequests: make(map[string]int64),
		stopChan:       make(chan struct{}),
	}, nil
}

// Start begins the background metrics computation goroutine
func (m *WindowedMetricsManager) Start() {
	if m.running {
		return
	}
	m.running = true

	go func() {
		ticker := time.NewTicker(m.updateInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.computeWindowedMetrics()
			case <-m.stopChan:
				return
			}
		}
	}()
}

// Stop stops the background metrics computation
func (m *WindowedMetricsManager) Stop() {
	if !m.running {
		return
	}
	close(m.stopChan)
	m.running = false
}

// RecordRequest records a request for windowed metrics tracking
func (m *WindowedMetricsManager) RecordRequest(req RequestData) {
	if !m.config.Enabled {
		return
	}

	key := req.Model

	m.bufferMutex.Lock()
	buffer, exists := m.requestBuffers[key]
	if !exists {
		// Check if we've hit max models
		if len(m.requestBuffers) >= m.maxModels {
			m.bufferMutex.Unlock()
			return
		}
		buffer = NewRequestRingBuffer(10000) // Adjust capacity as needed
		m.requestBuffers[key] = buffer
	}
	m.bufferMutex.Unlock()

	buffer.Add(req)
}

// IncrementActiveRequests increments the active request count for queue depth tracking
func (m *WindowedMetricsManager) IncrementActiveRequests(model string) {
	if !m.config.QueueDepthEstimation {
		return
	}

	key := model

	m.activeMutex.Lock()
	m.activeRequests[key]++
	count := m.activeRequests[key]
	m.activeMutex.Unlock()

	// Update the gauge immediately (only if metrics are initialized)
	if ModelQueueDepth == nil {
		return
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	ModelQueueDepth.WithLabelValues(model).Set(float64(count))
}

// DecrementActiveRequests decrements the active request count
func (m *WindowedMetricsManager) DecrementActiveRequests(model string) {
	if !m.config.QueueDepthEstimation {
		return
	}

	key := model

	m.activeMutex.Lock()
	m.activeRequests[key]--
	if m.activeRequests[key] < 0 {
		m.activeRequests[key] = 0
	}
	count := m.activeRequests[key]
	m.activeMutex.Unlock()

	// Update the gauge immediately (only if metrics are initialized)
	if ModelQueueDepth == nil {
		return
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	ModelQueueDepth.WithLabelValues(model).Set(float64(count))
}

// computeWindowedMetrics computes all windowed metrics
func (m *WindowedMetricsManager) computeWindowedMetrics() {
	now := time.Now()

	m.bufferMutex.RLock()
	buffers := make(map[string]*RequestRingBuffer, len(m.requestBuffers))
	for k, v := range m.requestBuffers {
		buffers[k] = v
	}
	m.bufferMutex.RUnlock()

	// Compute metrics for each model and each time window
	for model, buffer := range buffers {
		if model == "" {
			model = consts.UnknownLabel
		}

		for i, window := range m.timeWindows {
			windowLabel := m.windowLabels[i]
			since := now.Add(-window)
			data := buffer.GetDataSince(since)

			if len(data) == 0 {
				// Set zero values for empty windows
				ModelRequestsWindowed.WithLabelValues(model, windowLabel).Set(0)
				ModelLatencyWindowed.WithLabelValues(model, windowLabel).Set(0)
				ModelErrorRate.WithLabelValues(model, windowLabel).Set(0)
				continue
			}

			// Compute metrics
			var totalLatency float64
			var totalPromptTokens, totalCompletionTokens int64
			var errorCount int
			latencies := make([]float64, 0, len(data))

			for _, d := range data {
				totalLatency += d.LatencySeconds
				totalPromptTokens += d.PromptTokens
				totalCompletionTokens += d.CompletionTokens
				latencies = append(latencies, d.LatencySeconds)
				if d.IsError || d.IsTimeout {
					errorCount++
				}
			}

			requestCount := float64(len(data))
			avgLatency := totalLatency / requestCount
			errorRate := float64(errorCount) / requestCount

			// Update Prometheus metrics
			ModelRequestsWindowed.WithLabelValues(model, windowLabel).Set(requestCount)
			ModelLatencyWindowed.WithLabelValues(model, windowLabel).Set(avgLatency)
			ModelTokensWindowed.WithLabelValues(model, "prompt", windowLabel).Set(float64(totalPromptTokens))
			ModelTokensWindowed.WithLabelValues(model, "completion", windowLabel).Set(float64(totalCompletionTokens))
			ModelErrorRate.WithLabelValues(model, windowLabel).Set(errorRate)

			// Compute percentiles
			if len(latencies) > 0 {
				p50 := computePercentile(latencies, 0.50)
				p95 := computePercentile(latencies, 0.95)
				p99 := computePercentile(latencies, 0.99)

				ModelLatencyP50.WithLabelValues(model, windowLabel).Set(p50)
				ModelLatencyP95.WithLabelValues(model, windowLabel).Set(p95)
				ModelLatencyP99.WithLabelValues(model, windowLabel).Set(p99)
			}

			// Compute utilization (requests per second / expected capacity)
			// This is a simple approximation based on request rate
			requestsPerSecond := requestCount / window.Seconds()
			// Assume 100 req/s as theoretical max for utilization calculation
			// This can be made configurable
			utilization := (requestsPerSecond / 100.0) * 100.0
			if utilization > 100.0 {
				utilization = 100.0
			}
			ModelUtilization.WithLabelValues(model, windowLabel).Set(utilization)
		}
	}
}

// computePercentile computes the given percentile from a slice of values
func computePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort the values
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sortFloat64s(sorted)

	// Calculate the index
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
	// Simple insertion sort for small slices, quick sort for larger
	if len(a) < 12 {
		for i := 1; i < len(a); i++ {
			for j := i; j > 0 && a[j] < a[j-1]; j-- {
				a[j], a[j-1] = a[j-1], a[j]
			}
		}
		return
	}

	// Quick sort
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

// Global helper functions for recording windowed metrics

// RecordModelWindowedRequest records a request to the global windowed metrics manager
func RecordModelWindowedRequest(model string, latencySeconds float64, promptTokens, completionTokens int64, isError, isTimeout bool) {
	globalWindowedManagerMutex.RLock()
	manager := globalWindowedManager
	globalWindowedManagerMutex.RUnlock()

	if manager == nil {
		return
	}

	manager.RecordRequest(RequestData{
		Timestamp:        time.Now(),
		Model:            model,
		LatencySeconds:   latencySeconds,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		IsError:          isError,
		IsTimeout:        isTimeout,
	})
}

// IncrementModelActiveRequests increments the active request count
func IncrementModelActiveRequests(model string) {
	globalWindowedManagerMutex.RLock()
	manager := globalWindowedManager
	globalWindowedManagerMutex.RUnlock()

	if manager == nil {
		return
	}

	manager.IncrementActiveRequests(model)
}

// DecrementModelActiveRequests decrements the active request count
func DecrementModelActiveRequests(model string) {
	globalWindowedManagerMutex.RLock()
	manager := globalWindowedManager
	globalWindowedManagerMutex.RUnlock()

	if manager == nil {
		return
	}

	manager.DecrementActiveRequests(model)
}

// GetWindowedMetricsManager returns the global windowed metrics manager
func GetWindowedMetricsManager() *WindowedMetricsManager {
	globalWindowedManagerMutex.RLock()
	defer globalWindowedManagerMutex.RUnlock()
	return globalWindowedManager
}

// IsWindowedMetricsEnabled returns true if windowed metrics are enabled
func IsWindowedMetricsEnabled() bool {
	globalWindowedManagerMutex.RLock()
	manager := globalWindowedManager
	globalWindowedManagerMutex.RUnlock()

	return manager != nil && manager.config.Enabled
}
