package scoring

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Global scorer instance
var (
	globalScorer      *DynamicScorer
	globalScorerMutex sync.RWMutex
)

// InitializeDynamicScoring initializes the global dynamic scoring system
func InitializeDynamicScoring(cfg config.DynamicScoringConfig) error {
	if !cfg.Enabled {
		logging.Infof("Dynamic scoring is disabled")
		return nil
	}

	// Initialize metrics provider first
	maxModels := 100
	InitializeMetricsProvider(maxModels)
	provider := GetMetricsProvider()

	// Convert config types to scoring types
	scoringConfig := DynamicScoringConfig{
		Enabled:        cfg.Enabled,
		TimeWindow:     cfg.TimeWindow,
		UpdateInterval: cfg.UpdateInterval,
		MinSamples:     cfg.MinSamples,
		DefaultScore:   cfg.DefaultScore,
		DecayFactor:    cfg.DecayFactor,
		Weights: ScoreWeights{
			Accuracy: cfg.Weights.Accuracy,
			Latency:  cfg.Weights.Latency,
			Cost:     cfg.Weights.Cost,
		},
		Normalization: NormalizationConfig{
			LatencyTarget: cfg.Normalization.LatencyTarget,
			LatencyMax:    cfg.Normalization.LatencyMax,
			CostTarget:    cfg.Normalization.CostTarget,
			CostMax:       cfg.Normalization.CostMax,
		},
	}

	scorer, err := NewDynamicScorer(scoringConfig, provider)
	if err != nil {
		return err
	}

	globalScorerMutex.Lock()
	globalScorer = scorer
	globalScorerMutex.Unlock()

	scorer.Start()
	logging.Infof("Dynamic scoring initialized and started")

	return nil
}

// GetDynamicScorer returns the global dynamic scorer
func GetDynamicScorer() *DynamicScorer {
	globalScorerMutex.RLock()
	defer globalScorerMutex.RUnlock()
	return globalScorer
}

// IsDynamicScoringEnabled returns whether dynamic scoring is enabled and running
func IsDynamicScoringEnabled() bool {
	globalScorerMutex.RLock()
	scorer := globalScorer
	globalScorerMutex.RUnlock()
	return scorer != nil && scorer.IsEnabled()
}

// GetDynamicScore returns the dynamic score for a model
// Returns the default score if scoring is not enabled or model is unknown
func GetDynamicScore(model string) float64 {
	globalScorerMutex.RLock()
	scorer := globalScorer
	globalScorerMutex.RUnlock()

	if scorer == nil {
		return 0.5 // Default score
	}

	score, _ := scorer.GetScore(model)
	return score
}

// SelectBestModelByScore returns the model with the highest dynamic score from candidates
func SelectBestModelByScore(candidates []string) (string, float64) {
	globalScorerMutex.RLock()
	scorer := globalScorer
	globalScorerMutex.RUnlock()

	if scorer == nil || len(candidates) == 0 {
		if len(candidates) > 0 {
			return candidates[0], 0.5
		}
		return "", 0
	}

	return scorer.GetBestModel(candidates)
}

// StopDynamicScoring stops the global scoring system
func StopDynamicScoring() {
	globalScorerMutex.Lock()
	defer globalScorerMutex.Unlock()

	if globalScorer != nil {
		globalScorer.Stop()
		globalScorer = nil
	}
}
