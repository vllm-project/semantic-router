package latency

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// TPOTAlpha is the exponential moving average weight for TPOT smoothing
// 0.3 means: 30% new value, 70% historical average
const TPOTAlpha = 0.3

// MaxTPOTHistorySize limits the number of recent TPOT values stored per model
// This prevents unbounded memory growth while providing enough data for percentile calculation
const MaxTPOTHistorySize = 1000

// MinObservationsForPercentile is the minimum number of observations for reliable percentile calculation
// For 1-2 observations, we use the average value
// For 3+ observations, we use percentile calculation
const MinObservationsForPercentile = 3

// TPOTCache stores recent TPOT values per model for latency_aware percentile-based model selection.
// Every request updates it, under a plain mutex: the critical sections are a
// few hundred nanoseconds, and an RWMutex writer waiting on readers would
// stall every request behind it.
type TPOTCache struct {
	mu    sync.Mutex
	cache map[string]*ModelTPOTStats
}

// ModelTPOTStats stores TPOT statistics for a model
type ModelTPOTStats struct {
	LastTPOT         float64   // Most recent TPOT value
	AverageTPOT      float64   // Average TPOT over recent observations
	LastUpdated      time.Time // Last time TPOT was updated
	ObservationCount int       // Number of observations
	recent           window    // Last MaxTPOTHistorySize values, for percentiles
}

// Global TPOT cache instance
var globalTPOTCache = &TPOTCache{
	cache: make(map[string]*ModelTPOTStats),
}

// UpdateTPOT updates the TPOT cache for a model
func UpdateTPOT(model string, tpot float64) {
	// Normalize model name
	model = strings.TrimSpace(model)

	// Validate input: model name must not be empty, TPOT must be positive and within reasonable bounds
	// Note: TPOT cannot be negative in reality (time cannot be negative), but we validate for safety
	const minTPOT = 0.0001 // 0.1ms - very fast but realistic lower bound
	const maxTPOT = 1000.0 // 1000s - very slow but possible upper bound
	if model == "" {
		logging.Debugf("UpdateTPOT: skipping invalid input (empty model name)")
		return
	}
	if tpot <= 0 {
		logging.Debugf("UpdateTPOT: skipping invalid input (model=%q, tpot=%.4f - must be positive)", model, tpot)
		return
	}
	if tpot < minTPOT || tpot > maxTPOT {
		logging.Warnf("UpdateTPOT: suspicious TPOT value (model=%q, tpot=%.4f - outside normal range [%.4f, %.4f])", model, tpot, minTPOT, maxTPOT)
		// Still record it, but log a warning
	}

	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()

	stats, exists := globalTPOTCache.cache[model]
	if !exists {
		stats = &ModelTPOTStats{
			LastTPOT:         tpot,
			AverageTPOT:      tpot,
			LastUpdated:      time.Now(),
			ObservationCount: 1,
			recent:           newWindow(tpot),
		}
		globalTPOTCache.cache[model] = stats
	} else {
		// Update with exponential moving average
		// Formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
		stats.AverageTPOT = TPOTAlpha*tpot + (1-TPOTAlpha)*stats.AverageTPOT
		stats.LastTPOT = tpot
		stats.LastUpdated = time.Now()
		stats.ObservationCount++
		stats.recent.add(tpot, MaxTPOTHistorySize)
	}
}

// GetTPOT retrieves the current TPOT value for a model
func GetTPOT(model string) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, false
	}

	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()

	stats, exists := globalTPOTCache.cache[model]
	if !exists {
		return 0, false
	}

	// Use average TPOT if available, otherwise use last TPOT
	if stats.AverageTPOT > 0 {
		return stats.AverageTPOT, true
	}
	return stats.LastTPOT, true
}

// GetTPOTPercentile retrieves the percentile value for a model's TPOT distribution
// percentile should be between 1 and 100 (e.g., 10 for 10th percentile, 50 for median)
// Returns the TPOT value at the specified percentile and whether data exists
// Works with any number of observations (1+): uses average for 1-2, percentile for 3+
func GetTPOTPercentile(model string, percentile int) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" || percentile < 1 || percentile > 100 {
		return 0, false
	}

	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()

	stats, exists := globalTPOTCache.cache[model]
	if !exists || stats.recent.len() == 0 {
		return 0, false
	}

	// For 1-2 observations, use average as threshold
	if stats.recent.len() < MinObservationsForPercentile {
		// Use average TPOT as threshold for small sample sizes
		if stats.AverageTPOT > 0 {
			return stats.AverageTPOT, true
		}
		// Fallback to last TPOT if average not available
		return stats.LastTPOT, true
	}

	// For 3+ observations, read the percentile from the window's sorted values
	return percentileFromSorted(stats.recent.sorted, float64(percentile)/100.0)
}

// ResetTPOT clears the TPOT cache (useful for testing)
func ResetTPOT() {
	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()
	globalTPOTCache.cache = make(map[string]*ModelTPOTStats)
}

// TTFTAlpha is the exponential moving average weight for TTFT smoothing
// 0.3 means: 30% new value, 70% historical average
const TTFTAlpha = 0.3

// MaxTTFTHistorySize limits the number of recent TTFT values stored per model
// This prevents unbounded memory growth while providing enough data for percentile calculation
const MaxTTFTHistorySize = 1000

// TTFTCache stores recent TTFT values per model for latency_aware percentile-based model selection.
// Every request updates and reads it, under a plain mutex, as TPOTCache.
type TTFTCache struct {
	mu    sync.Mutex
	cache map[string]*ModelTTFTStats
}

// ModelTTFTStats stores TTFT statistics for a model
type ModelTTFTStats struct {
	LastTTFT         float64   // Most recent TTFT value
	AverageTTFT      float64   // Average TTFT over recent observations
	LastUpdated      time.Time // Last time TTFT was updated
	ObservationCount int       // Number of observations
	recent           window    // Last MaxTTFTHistorySize values, for percentiles
}

// Global TTFT cache instance
var globalTTFTCache = &TTFTCache{
	cache: make(map[string]*ModelTTFTStats),
}

// UpdateTTFT updates the TTFT cache for a model
func UpdateTTFT(model string, ttft float64) {
	// Normalize model name
	model = strings.TrimSpace(model)

	// Validate input: model name must not be empty, TTFT must be positive and within reasonable bounds
	// Note: TTFT cannot be negative in reality (time cannot be negative), but we validate for safety
	const minTTFT = 0.0001 // 0.1ms - very fast but realistic lower bound
	const maxTTFT = 1000.0 // 1000s - very slow but possible upper bound
	if model == "" {
		logging.Debugf("UpdateTTFT: skipping invalid input (empty model name)")
		return
	}
	if ttft <= 0 {
		logging.Debugf("UpdateTTFT: skipping invalid input (model=%q, ttft=%.4f - must be positive)", model, ttft)
		return
	}
	if ttft < minTTFT || ttft > maxTTFT {
		logging.Warnf("UpdateTTFT: suspicious TTFT value (model=%q, ttft=%.4f - outside normal range [%.4f, %.4f])", model, ttft, minTTFT, maxTTFT)
		// Still record it, but log a warning
	}

	globalTTFTCache.mu.Lock()
	defer globalTTFTCache.mu.Unlock()

	stats, exists := globalTTFTCache.cache[model]
	if !exists {
		stats = &ModelTTFTStats{
			LastTTFT:         ttft,
			AverageTTFT:      ttft,
			LastUpdated:      time.Now(),
			ObservationCount: 1,
			recent:           newWindow(ttft),
		}
		globalTTFTCache.cache[model] = stats
	} else {
		// Update with exponential moving average
		// Formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
		stats.AverageTTFT = TTFTAlpha*ttft + (1-TTFTAlpha)*stats.AverageTTFT
		stats.LastTTFT = ttft
		stats.LastUpdated = time.Now()
		stats.ObservationCount++
		stats.recent.add(ttft, MaxTTFTHistorySize)
	}
}

// GetTTFT retrieves the current TTFT value for a model
func GetTTFT(model string) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, false
	}

	globalTTFTCache.mu.Lock()
	defer globalTTFTCache.mu.Unlock()

	stats, exists := globalTTFTCache.cache[model]
	if !exists {
		return 0, false
	}

	// Use average TTFT if available, otherwise use last TTFT
	if stats.AverageTTFT > 0 {
		return stats.AverageTTFT, true
	}
	return stats.LastTTFT, true
}

// GetTTFTPercentile retrieves the percentile value for a model's TTFT distribution
// percentile should be between 1 and 100 (e.g., 10 for 10th percentile, 50 for median)
// Returns the TTFT value at the specified percentile and whether data exists
// Works with any number of observations (1+): uses average for 1-2, percentile for 3+
func GetTTFTPercentile(model string, percentile int) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" || percentile < 1 || percentile > 100 {
		return 0, false
	}

	globalTTFTCache.mu.Lock()
	defer globalTTFTCache.mu.Unlock()

	stats, exists := globalTTFTCache.cache[model]
	if !exists || stats.recent.len() == 0 {
		return 0, false
	}

	// For 1-2 observations, use average as threshold
	if stats.recent.len() < MinObservationsForPercentile {
		// Use average TTFT as threshold for small sample sizes
		if stats.AverageTTFT > 0 {
			return stats.AverageTTFT, true
		}
		// Fallback to last TTFT if average not available
		return stats.LastTTFT, true
	}

	// For 3+ observations, read the percentile from the window's sorted values
	return percentileFromSorted(stats.recent.sorted, float64(percentile)/100.0)
}

// ResetTTFT clears the TTFT cache (useful for testing)
func ResetTTFT() {
	globalTTFTCache.mu.Lock()
	defer globalTTFTCache.mu.Unlock()
	globalTTFTCache.cache = make(map[string]*ModelTTFTStats)
}

// RemoveModelFromTPOTCache removes a model from the TPOT cache
// This should be called when a model is removed from the system to prevent memory leaks
func RemoveModelFromTPOTCache(model string) {
	model = strings.TrimSpace(model)
	if model == "" {
		return
	}

	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()
	delete(globalTPOTCache.cache, model)
	logging.Debugf("Removed model %q from TPOT cache", model)
}

// RemoveModelFromTTFTCache removes a model from the TTFT cache
// This should be called when a model is removed from the system to prevent memory leaks
func RemoveModelFromTTFTCache(model string) {
	model = strings.TrimSpace(model)
	if model == "" {
		return
	}

	globalTTFTCache.mu.Lock()
	defer globalTTFTCache.mu.Unlock()
	delete(globalTTFTCache.cache, model)
	logging.Debugf("Removed model %q from TTFT cache", model)
}

// RemoveModelFromLatencyCache removes a model from both TPOT and TTFT caches
// This is a convenience function that calls both RemoveModelFromTPOTCache and RemoveModelFromTTFTCache
func RemoveModelFromLatencyCache(model string) {
	RemoveModelFromTPOTCache(model)
	RemoveModelFromTTFTCache(model)
}
