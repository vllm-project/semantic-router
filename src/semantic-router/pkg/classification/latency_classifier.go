package classification

import (
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// TPOTCache stores recent TPOT values per model for latency signal evaluation
type TPOTCache struct {
	mu    sync.RWMutex
	cache map[string]*ModelTPOTStats
}

// ModelTPOTStats stores TPOT statistics for a model
type ModelTPOTStats struct {
	LastTPOT         float64   // Most recent TPOT value
	AverageTPOT      float64   // Average TPOT over recent observations
	RecentTPOTs      []float64 // Recent TPOT values for percentile calculation (sliding window)
	LastUpdated      time.Time // Last time TPOT was updated
	ObservationCount int       // Number of observations
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
			RecentTPOTs:      []float64{tpot},
			LastUpdated:      time.Now(),
			ObservationCount: 1,
		}
		globalTPOTCache.cache[model] = stats
	} else {
		// Update with exponential moving average
		// Formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
		stats.AverageTPOT = TPOTAlpha*tpot + (1-TPOTAlpha)*stats.AverageTPOT
		stats.LastTPOT = tpot
		stats.LastUpdated = time.Now()
		stats.ObservationCount++

		// Add to recent TPOT history for percentile calculation
		stats.RecentTPOTs = append(stats.RecentTPOTs, tpot)

		// Maintain sliding window: keep only last MaxTPOTHistorySize values
		if len(stats.RecentTPOTs) > MaxTPOTHistorySize {
			// Remove oldest values, keeping the most recent ones
			stats.RecentTPOTs = stats.RecentTPOTs[len(stats.RecentTPOTs)-MaxTPOTHistorySize:]
		}
	}
}

// GetTPOT retrieves the current TPOT value for a model
func GetTPOT(model string) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, false
	}

	globalTPOTCache.mu.RLock()
	defer globalTPOTCache.mu.RUnlock()

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

	globalTPOTCache.mu.RLock()
	stats, exists := globalTPOTCache.cache[model]
	if !exists || len(stats.RecentTPOTs) == 0 {
		globalTPOTCache.mu.RUnlock()
		return 0, false
	}

	// Copy the slice while holding the lock to avoid race conditions with concurrent updates
	// This ensures we work with a consistent snapshot of the data
	recentTPOTs := make([]float64, len(stats.RecentTPOTs))
	copy(recentTPOTs, stats.RecentTPOTs)
	avg := stats.AverageTPOT
	last := stats.LastTPOT
	globalTPOTCache.mu.RUnlock()

	// For 1-2 observations, use average as threshold
	if len(recentTPOTs) < MinObservationsForPercentile {
		// Use average TPOT as threshold for small sample sizes
		if avg > 0 {
			return avg, true
		}
		// Fallback to last TPOT if average not available
		return last, true
	}

	// For 3+ observations, use proper percentile calculation
	// Note: Sorting is O(n log n), but necessary for accurate percentile calculation
	// With MaxTPOTHistorySize=1000, this is acceptable performance (~10k comparisons)
	percentileValue := computePercentile(recentTPOTs, float64(percentile)/100.0)
	return percentileValue, true
}

// computePercentile computes the given percentile from a slice of values
// percentile should be between 0.0 and 1.0 (e.g., 0.1 for 10th percentile, 0.5 for median)
// Note: This function assumes the input slice is already a copy (not shared with other goroutines)
// The caller is responsible for copying the slice while holding appropriate locks
// Performance: O(n log n) where n is len(values)
// With MaxTPOTHistorySize/MaxTTFTHistorySize=1000, this is ~10,000 comparisons
// This is acceptable for routing decisions, but if performance becomes an issue,
// consider: (1) maintaining sorted arrays, (2) approximate percentiles, or (3) caching results
func computePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort the values
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

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

// TTFTCache stores recent TTFT values per model for latency signal evaluation
type TTFTCache struct {
	mu    sync.RWMutex
	cache map[string]*ModelTTFTStats
}

// ModelTTFTStats stores TTFT statistics for a model
type ModelTTFTStats struct {
	LastTTFT         float64   // Most recent TTFT value
	AverageTTFT      float64   // Average TTFT over recent observations
	RecentTTFTs      []float64 // Recent TTFT values for percentile calculation (sliding window)
	LastUpdated      time.Time // Last time TTFT was updated
	ObservationCount int       // Number of observations
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
			RecentTTFTs:      []float64{ttft},
			LastUpdated:      time.Now(),
			ObservationCount: 1,
		}
		globalTTFTCache.cache[model] = stats
	} else {
		// Update with exponential moving average
		// Formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
		stats.AverageTTFT = TTFTAlpha*ttft + (1-TTFTAlpha)*stats.AverageTTFT
		stats.LastTTFT = ttft
		stats.LastUpdated = time.Now()
		stats.ObservationCount++

		// Add to recent TTFT history for percentile calculation
		stats.RecentTTFTs = append(stats.RecentTTFTs, ttft)

		// Maintain sliding window: keep only last MaxTTFTHistorySize values
		if len(stats.RecentTTFTs) > MaxTTFTHistorySize {
			// Remove oldest values, keeping the most recent ones
			stats.RecentTTFTs = stats.RecentTTFTs[len(stats.RecentTTFTs)-MaxTTFTHistorySize:]
		}
	}
}

// GetTTFT retrieves the current TTFT value for a model
func GetTTFT(model string) (float64, bool) {
	// Normalize model name
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, false
	}

	globalTTFTCache.mu.RLock()
	defer globalTTFTCache.mu.RUnlock()

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

	globalTTFTCache.mu.RLock()
	stats, exists := globalTTFTCache.cache[model]
	if !exists || len(stats.RecentTTFTs) == 0 {
		globalTTFTCache.mu.RUnlock()
		return 0, false
	}

	// Copy the slice while holding the lock to avoid race conditions with concurrent updates
	// This ensures we work with a consistent snapshot of the data
	recentTTFTs := make([]float64, len(stats.RecentTTFTs))
	copy(recentTTFTs, stats.RecentTTFTs)
	avg := stats.AverageTTFT
	last := stats.LastTTFT
	globalTTFTCache.mu.RUnlock()

	// For 1-2 observations, use average as threshold
	if len(recentTTFTs) < MinObservationsForPercentile {
		// Use average TTFT as threshold for small sample sizes
		if avg > 0 {
			return avg, true
		}
		// Fallback to last TTFT if average not available
		return last, true
	}

	// For 3+ observations, use proper percentile calculation
	// Note: Sorting is O(n log n), but necessary for accurate percentile calculation
	// With MaxTTFTHistorySize=1000, this is acceptable performance (~10k comparisons)
	percentileValue := computePercentile(recentTTFTs, float64(percentile)/100.0)
	return percentileValue, true
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

// LatencyClassifier implements latency-based signal classification using TPOT
// Evaluates whether models meet latency requirements based on their TPOT (Time Per Output Token)
type LatencyClassifier struct {
	rules []config.LatencyRule
}

// LatencyResult represents the result of latency classification
type LatencyResult struct {
	MatchedRules []string // Names of latency rules that matched
}

// NewLatencyClassifier creates a new latency classifier
func NewLatencyClassifier(cfgRules []config.LatencyRule) (*LatencyClassifier, error) {
	return &LatencyClassifier{
		rules: cfgRules,
	}, nil
}

// Classify evaluates latency rules against available models
// It checks if models in the decision's ModelRefs meet the latency requirements
// Uses percentile-based thresholds for TPOT and/or TTFT to determine matches
func (c *LatencyClassifier) Classify(availableModels []string) (*LatencyResult, error) {
	if len(c.rules) == 0 {
		return &LatencyResult{
			MatchedRules: []string{},
		}, nil
	}

	var matchedRules []string

	for _, rule := range c.rules {
		// Check if any available model meets this latency rule
		matched := false
		var bestTPOT float64
		var bestTTFT float64
		bestModel := ""
		bestScore := -1.0 // Initialize to -1 to indicate "no match yet"

		// Determine which metrics to evaluate
		hasTPOTPercentile := rule.TPOTPercentile > 0
		hasTTFTPercentile := rule.TTFTPercentile > 0

		if !hasTPOTPercentile && !hasTTFTPercentile {
			// Invalid rule configuration (should be caught by validator, but handle gracefully)
			logging.Warnf("Latency rule '%s' has neither tpot_percentile nor ttft_percentile set, skipping", rule.Name)
			continue
		}

		for _, model := range availableModels {
			// Normalize model name to ensure consistent matching with cache keys
			// (UpdateTPOT/UpdateTTFT normalize model names, so we must normalize here too)
			model = strings.TrimSpace(model)
			if model == "" {
				logging.Debugf("Latency evaluation: skipping empty model name")
				continue
			}

			// Evaluate TPOT if configured
			tpot := 0.0
			hasTPOT := false
			tpotMeetsThreshold := true
			tpotThreshold := 0.0
			if hasTPOTPercentile {
				// Use median (50th percentile) as "current" so a few slow requests don't pull average up and stop matching
				currentTPOT, tpotExists := GetTPOTPercentile(model, 50)
				if !tpotExists {
					logging.Infof("[Latency] No TPOT data for model %q, skipping rule %s", model, rule.Name)
					continue
				}
				tpot = currentTPOT
				hasTPOT = true

				// Get TPOT percentile threshold
				tpotThresholdValue, hasTPOTThreshold := GetTPOTPercentile(model, rule.TPOTPercentile)
				if !hasTPOTThreshold {
					logging.Debugf("Latency evaluation: cannot calculate TPOT percentile %d for model %q, skipping", rule.TPOTPercentile, model)
					continue
				}
				tpotThreshold = tpotThresholdValue
				tpotMeetsThreshold = tpot <= tpotThreshold
			}

			// Evaluate TTFT if configured
			// Use GetTTFT() function for consistent access pattern (same as TPOT)
			ttft := 0.0
			hasTTFT := false
			ttftMeetsThreshold := true
			ttftThreshold := 0.0
			if hasTTFTPercentile {
				// Use median (50th percentile) as "current" so a few slow requests don't pull average up and stop matching
				currentTTFT, ttftExists := GetTTFTPercentile(model, 50)
				if !ttftExists {
					logging.Infof("[Latency] No TTFT data for model %q, skipping rule %s", model, rule.Name)
					continue
				}
				ttft = currentTTFT
				hasTTFT = true

				// Get TTFT percentile threshold
				ttftThresholdValue, hasTTFTThreshold := GetTTFTPercentile(model, rule.TTFTPercentile)
				if !hasTTFTThreshold {
					logging.Debugf("Latency evaluation: cannot calculate TTFT percentile %d for model %q, skipping", rule.TTFTPercentile, model)
					continue
				}
				ttftThreshold = ttftThresholdValue
				ttftMeetsThreshold = ttft <= ttftThreshold
			}

			// Model matches if it meets all configured thresholds (AND logic)
			// If only one metric is configured, only that metric needs to meet threshold
			//
			// Percentile semantics (lower latency is better): The Nth percentile is the value below which N%
			// of observations fall (the "top N% fastest" boundary). We check current ≤ threshold, so
			// "current ≤ 10th percentile" means current is in the top 10% fastest; "current ≤ 70th percentile"
			// means current is in the faster 70% (more lenient). Higher N = easier to match.
			matchesRule := true
			if hasTPOTPercentile && !tpotMeetsThreshold {
				matchesRule = false
			}
			if hasTTFTPercentile && !ttftMeetsThreshold {
				matchesRule = false
			}

			if matchesRule {
				// Calculate combined score (lower is better) for selecting best model
				// Normalize by threshold to make TPOT and TTFT comparable
				var combinedScore float64
				// Safety check: prevent division by zero (threshold should never be 0, but protect against edge cases)
				if hasTPOT && hasTTFT {
					// Normalize both metrics (use ratio to threshold) and combine
					if tpotThreshold > 0 && ttftThreshold > 0 {
						tpotRatio := tpot / tpotThreshold
						ttftRatio := ttft / ttftThreshold
						// Use average: both metrics have equal weight
						combinedScore = (tpotRatio + ttftRatio) / 2.0
					} else {
						logging.Warnf("Latency evaluation: zero threshold detected (TPOT=%.4f, TTFT=%.4f), skipping score calculation", tpotThreshold, ttftThreshold)
						continue
					}
				} else if hasTPOT {
					if tpotThreshold > 0 {
						combinedScore = tpot / tpotThreshold
					} else {
						logging.Warnf("Latency evaluation: zero TPOT threshold detected, skipping score calculation")
						continue
					}
				} else if hasTTFT {
					if ttftThreshold > 0 {
						combinedScore = ttft / ttftThreshold
					} else {
						logging.Warnf("Latency evaluation: zero TTFT threshold detected, skipping score calculation")
						continue
					}
				}

				// Select best model: lower score = better (closer to threshold = better performance)
				if !matched || combinedScore < bestScore {
					matched = true
					bestTPOT = tpot
					bestTTFT = ttft
					bestModel = model
					bestScore = combinedScore
				}
			}
		}

		if matched {
			matchedRules = append(matchedRules, rule.Name)

			// Log the match (no confidence - latency matching is deterministic)
			if hasTPOTPercentile && hasTTFTPercentile {
				// Get thresholds for logging (recalculate if needed)
				tpotThreshold, _ := GetTPOTPercentile(bestModel, rule.TPOTPercentile)
				ttftThreshold, _ := GetTTFTPercentile(bestModel, rule.TTFTPercentile)
				logging.Infof("Latency rule '%s' matched: model=%s, TPOT=%.4fs (threshold=%.4fs, percentile=%d), TTFT=%.4fs (threshold=%.4fs, percentile=%d), score=%.4f",
					rule.Name, bestModel, bestTPOT, tpotThreshold, rule.TPOTPercentile, bestTTFT, ttftThreshold, rule.TTFTPercentile, bestScore)
			} else if hasTPOTPercentile {
				tpotThreshold, _ := GetTPOTPercentile(bestModel, rule.TPOTPercentile)
				logging.Infof("Latency rule '%s' matched: model=%s, TPOT=%.4fs (threshold=%.4fs, percentile=%d), score=%.4f",
					rule.Name, bestModel, bestTPOT, tpotThreshold, rule.TPOTPercentile, bestScore)
			} else if hasTTFTPercentile {
				ttftThreshold, _ := GetTTFTPercentile(bestModel, rule.TTFTPercentile)
				logging.Infof("Latency rule '%s' matched: model=%s, TTFT=%.4fs (threshold=%.4fs, percentile=%d), score=%.4f",
					rule.Name, bestModel, bestTTFT, ttftThreshold, rule.TTFTPercentile, bestScore)
			}
		}
	}

	return &LatencyResult{
		MatchedRules: matchedRules,
	}, nil
}
