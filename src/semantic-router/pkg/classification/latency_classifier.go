package classification

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// TPOTAlpha is the exponential moving average weight for TPOT smoothing
// 0.3 means: 30% new value, 70% historical average
const TPOTAlpha = 0.3

// TPOTCache stores recent TPOT values per model for latency signal evaluation
type TPOTCache struct {
	mu    sync.RWMutex
	cache map[string]*ModelTPOTStats
}

// ModelTPOTStats stores TPOT statistics for a model
type ModelTPOTStats struct {
	LastTPOT         float64   // Most recent TPOT value
	AverageTPOT      float64   // Average TPOT over recent observations
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
	if model == "" || tpot <= 0 {
		logging.Debugf("UpdateTPOT: skipping invalid input (model=%q, tpot=%.4f)", model, tpot)
		return
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
		}
		globalTPOTCache.cache[model] = stats
	} else {
		// Update with exponential moving average
		// Formula: new_avg = alpha * new_value + (1 - alpha) * old_avg
		stats.AverageTPOT = TPOTAlpha*tpot + (1-TPOTAlpha)*stats.AverageTPOT
		stats.LastTPOT = tpot
		stats.LastUpdated = time.Now()
		stats.ObservationCount++
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

// ResetTPOT clears the TPOT cache (useful for testing)
func ResetTPOT() {
	globalTPOTCache.mu.Lock()
	defer globalTPOTCache.mu.Unlock()
	globalTPOTCache.cache = make(map[string]*ModelTPOTStats)
}

// LatencyClassifier implements latency-based signal classification using TPOT
// Evaluates whether models meet latency requirements based on their TPOT (Time Per Output Token)
type LatencyClassifier struct {
	rules []config.LatencyRule
}

// LatencyResult represents the result of latency classification
type LatencyResult struct {
	MatchedRules []string // Names of latency rules that matched
	Confidence   float64  // Confidence score (0.0-1.0)
}

// NewLatencyClassifier creates a new latency classifier
func NewLatencyClassifier(cfgRules []config.LatencyRule) (*LatencyClassifier, error) {
	return &LatencyClassifier{
		rules: cfgRules,
	}, nil
}

// Classify evaluates latency rules against available models
// It checks if models in the decision's ModelRefs meet the latency requirements
func (c *LatencyClassifier) Classify(availableModels []string) (*LatencyResult, error) {
	if len(c.rules) == 0 {
		return &LatencyResult{
			MatchedRules: []string{},
			Confidence:   0.0,
		}, nil
	}

	var matchedRules []string
	totalConfidence := 0.0
	matchedCount := 0

	for _, rule := range c.rules {
		// Check if any available model meets this latency rule
		matched := false
		bestTPOT := 0.0
		bestModel := ""

		for _, model := range availableModels {
			tpot, exists := GetTPOT(model)
			if !exists {
				// No TPOT data for this model, skip
				logging.Infof("Latency evaluation: no TPOT data for model %q, skipping", model)
				continue
			}
			logging.Infof("Latency evaluation: model=%s, TPOT=%.4fs", model, tpot)

			// Check if model meets the latency threshold
			if rule.MaxTPOT > 0 && tpot <= rule.MaxTPOT {
				if !matched || tpot < bestTPOT {
					matched = true
					bestTPOT = tpot
					bestModel = model
				}
			}
		}

		if matched {
			matchedRules = append(matchedRules, rule.Name)

			// Calculate confidence based on how much better the TPOT is than the threshold
			// If TPOT is much lower than threshold, higher confidence
			if rule.MaxTPOT > 0 {
				ratio := bestTPOT / rule.MaxTPOT
				confidence := 1.0 - ratio // Lower ratio = higher confidence
				if confidence < 0.5 {
					confidence = 0.5 // Minimum confidence
				}
				if confidence > 1.0 {
					confidence = 1.0
				}
				totalConfidence += confidence
				matchedCount++

				logging.Infof("Latency rule '%s' matched: model=%s, TPOT=%.4fs, threshold=%.4fs, confidence=%.2f",
					rule.Name, bestModel, bestTPOT, rule.MaxTPOT, confidence)
			}
		}
	}

	// Calculate average confidence
	avgConfidence := 0.0
	if matchedCount > 0 {
		avgConfidence = totalConfidence / float64(matchedCount)
	}

	return &LatencyResult{
		MatchedRules: matchedRules,
		Confidence:   avgConfidence,
	}, nil
}
