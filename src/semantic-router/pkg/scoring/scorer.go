package scoring

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DynamicScoringConfig configures the dynamic model scoring system
type DynamicScoringConfig struct {
	// Enabled enables dynamic scoring
	Enabled bool `yaml:"enabled" json:"enabled"`

	// TimeWindow specifies the time window for score computation (e.g., "5m", "1h")
	// Default: "5m"
	TimeWindow string `yaml:"time_window,omitempty" json:"time_window,omitempty"`

	// UpdateInterval specifies how often scores are recomputed
	// Default: "30s"
	UpdateInterval string `yaml:"update_interval,omitempty" json:"update_interval,omitempty"`

	// Weights for combining component scores (must sum to 1.0)
	Weights ScoreWeights `yaml:"weights" json:"weights"`

	// Normalization settings for each component
	Normalization NormalizationConfig `yaml:"normalization" json:"normalization"`

	// MinSamples is the minimum number of samples required before dynamic scoring is used
	// Below this threshold, the default score is used
	// Default: 10
	MinSamples int `yaml:"min_samples,omitempty" json:"min_samples,omitempty"`

	// DefaultScore is used when there's insufficient data for a model
	// Default: 0.5
	DefaultScore float64 `yaml:"default_score,omitempty" json:"default_score,omitempty"`

	// DecayFactor for exponential moving average (0.0-1.0)
	// Higher values give more weight to recent data
	// Default: 0.1
	DecayFactor float64 `yaml:"decay_factor,omitempty" json:"decay_factor,omitempty"`
}

// ScoreWeights defines the weights for combining score components
type ScoreWeights struct {
	// Accuracy weight (based on success rate, 1 - error_rate)
	Accuracy float64 `yaml:"accuracy" json:"accuracy"`

	// Latency weight (lower is better, normalized)
	Latency float64 `yaml:"latency" json:"latency"`

	// Cost weight (lower is better, normalized)
	Cost float64 `yaml:"cost" json:"cost"`
}

// NormalizationConfig configures how raw metrics are normalized to 0-1 scale
type NormalizationConfig struct {
	// LatencyTarget is the target latency in seconds (scores 0.5 at target)
	// Default: 1.0
	LatencyTarget float64 `yaml:"latency_target,omitempty" json:"latency_target,omitempty"`

	// LatencyMax is the maximum acceptable latency (scores 0.0 at or above)
	// Default: 10.0
	LatencyMax float64 `yaml:"latency_max,omitempty" json:"latency_max,omitempty"`

	// CostTarget is the target cost per 1K tokens in USD (scores 0.5 at target)
	// Default: 0.01
	CostTarget float64 `yaml:"cost_target,omitempty" json:"cost_target,omitempty"`

	// CostMax is the maximum acceptable cost per 1K tokens (scores 0.0 at or above)
	// Default: 0.1
	CostMax float64 `yaml:"cost_max,omitempty" json:"cost_max,omitempty"`
}

// ModelScore represents the computed dynamic score for a model
type ModelScore struct {
	// Model is the model identifier
	Model string `json:"model"`

	// CompositeScore is the final weighted score (0.0-1.0)
	CompositeScore float64 `json:"composite_score"`

	// ComponentScores contains individual component scores
	ComponentScores ComponentScores `json:"component_scores"`

	// Metrics contains the raw metrics used for scoring
	Metrics ModelMetrics `json:"metrics"`

	// SampleCount is the number of samples used for this score
	SampleCount int `json:"sample_count"`

	// LastUpdated is when this score was last computed
	LastUpdated time.Time `json:"last_updated"`

	// IsDefault indicates if default score was used due to insufficient data
	IsDefault bool `json:"is_default"`
}

// ComponentScores contains individual normalized scores
type ComponentScores struct {
	Accuracy float64 `json:"accuracy"`
	Latency  float64 `json:"latency"`
	Cost     float64 `json:"cost"`
}

// ModelMetrics contains raw metrics for a model
type ModelMetrics struct {
	// RequestCount is the total number of requests
	RequestCount int `json:"request_count"`

	// ErrorCount is the number of failed requests
	ErrorCount int `json:"error_count"`

	// ErrorRate is the ratio of errors to total requests
	ErrorRate float64 `json:"error_rate"`

	// AvgLatencySeconds is the average response latency
	AvgLatencySeconds float64 `json:"avg_latency_seconds"`

	// P95LatencySeconds is the 95th percentile latency
	P95LatencySeconds float64 `json:"p95_latency_seconds"`

	// TotalTokens is the total tokens processed
	TotalTokens int64 `json:"total_tokens"`

	// TotalCost is the total cost in USD
	TotalCost float64 `json:"total_cost"`

	// CostPerKTokens is the cost per 1000 tokens
	CostPerKTokens float64 `json:"cost_per_k_tokens"`
}

// MetricsProvider is an interface for getting model metrics
type MetricsProvider interface {
	// GetModelMetrics returns metrics for a specific model over the given time window
	GetModelMetrics(model string, window time.Duration) (ModelMetrics, error)

	// GetAllModelMetrics returns metrics for all tracked models
	GetAllModelMetrics(window time.Duration) (map[string]ModelMetrics, error)
}

// Prometheus metrics for dynamic scoring
var (
	DynamicScoreGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_model_dynamic_score",
			Help: "Dynamic composite score for model selection (0.0-1.0)",
		},
		[]string{"model"},
	)

	AccuracyScoreGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_model_accuracy_score",
			Help: "Accuracy component score (1 - error_rate)",
		},
		[]string{"model"},
	)

	LatencyScoreGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_model_latency_score",
			Help: "Latency component score (normalized, lower latency is higher score)",
		},
		[]string{"model"},
	)

	CostScoreGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_model_cost_score",
			Help: "Cost component score (normalized, lower cost is higher score)",
		},
		[]string{"model"},
	)

	ScoringUpdateCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_scoring_updates_total",
			Help: "Number of scoring update cycles",
		},
		[]string{"status"},
	)
)

// DynamicScorer computes and maintains dynamic model scores
type DynamicScorer struct {
	config          DynamicScoringConfig
	metricsProvider MetricsProvider
	timeWindow      time.Duration
	updateInterval  time.Duration

	// Current scores for all models
	scores      map[string]*ModelScore
	scoresMutex sync.RWMutex

	// Exponential moving averages for score smoothing
	emaScores map[string]float64
	emaMutex  sync.RWMutex

	// Background update control
	stopChan     chan struct{}
	running      bool
	runningMutex sync.Mutex
}

// NewDynamicScorer creates a new DynamicScorer
func NewDynamicScorer(config DynamicScoringConfig, provider MetricsProvider) (*DynamicScorer, error) {
	// Parse time window
	timeWindow := 5 * time.Minute
	if config.TimeWindow != "" {
		if d, err := time.ParseDuration(config.TimeWindow); err == nil {
			timeWindow = d
		}
	}

	// Parse update interval
	updateInterval := 30 * time.Second
	if config.UpdateInterval != "" {
		if d, err := time.ParseDuration(config.UpdateInterval); err == nil {
			updateInterval = d
		}
	}

	// Apply defaults
	if config.MinSamples <= 0 {
		config.MinSamples = 10
	}
	if config.DefaultScore <= 0 {
		config.DefaultScore = 0.5
	}
	if config.DecayFactor <= 0 || config.DecayFactor > 1 {
		config.DecayFactor = 0.1
	}

	// Apply normalization defaults
	if config.Normalization.LatencyTarget <= 0 {
		config.Normalization.LatencyTarget = 1.0
	}
	if config.Normalization.LatencyMax <= 0 {
		config.Normalization.LatencyMax = 10.0
	}
	if config.Normalization.CostTarget <= 0 {
		config.Normalization.CostTarget = 0.01
	}
	if config.Normalization.CostMax <= 0 {
		config.Normalization.CostMax = 0.1
	}

	// Validate weights sum to 1.0
	weightsSum := config.Weights.Accuracy + config.Weights.Latency + config.Weights.Cost
	if weightsSum == 0 {
		// Default weights: equal distribution
		config.Weights.Accuracy = 0.5
		config.Weights.Latency = 0.3
		config.Weights.Cost = 0.2
	} else if math.Abs(weightsSum-1.0) > 0.01 {
		// Normalize weights to sum to 1.0
		config.Weights.Accuracy /= weightsSum
		config.Weights.Latency /= weightsSum
		config.Weights.Cost /= weightsSum
	}

	return &DynamicScorer{
		config:          config,
		metricsProvider: provider,
		timeWindow:      timeWindow,
		updateInterval:  updateInterval,
		scores:          make(map[string]*ModelScore),
		emaScores:       make(map[string]float64),
	}, nil
}

// Start begins the background score update goroutine
func (s *DynamicScorer) Start() {
	s.runningMutex.Lock()
	defer s.runningMutex.Unlock()

	if s.running || !s.config.Enabled {
		return
	}
	s.running = true
	s.stopChan = make(chan struct{})

	// Initial computation
	s.updateScores()

	go func() {
		ticker := time.NewTicker(s.updateInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				s.updateScores()
			case <-s.stopChan:
				return
			}
		}
	}()

	logging.Infof("Dynamic scoring started: window=%v, interval=%v, weights={accuracy:%.2f, latency:%.2f, cost:%.2f}",
		s.timeWindow, s.updateInterval, s.config.Weights.Accuracy, s.config.Weights.Latency, s.config.Weights.Cost)
}

// Stop stops the background update goroutine
func (s *DynamicScorer) Stop() {
	s.runningMutex.Lock()
	defer s.runningMutex.Unlock()

	if !s.running {
		return
	}
	close(s.stopChan)
	s.running = false
}

// GetScore returns the current score for a model
func (s *DynamicScorer) GetScore(model string) (float64, bool) {
	s.scoresMutex.RLock()
	defer s.scoresMutex.RUnlock()

	if score, exists := s.scores[model]; exists {
		return score.CompositeScore, true
	}
	return s.config.DefaultScore, false
}

// GetModelScore returns the full score details for a model
func (s *DynamicScorer) GetModelScore(model string) (*ModelScore, bool) {
	s.scoresMutex.RLock()
	defer s.scoresMutex.RUnlock()

	if score, exists := s.scores[model]; exists {
		// Return a copy to prevent modification
		scoreCopy := *score
		return &scoreCopy, true
	}
	return nil, false
}

// GetAllScores returns all current model scores
func (s *DynamicScorer) GetAllScores() map[string]*ModelScore {
	s.scoresMutex.RLock()
	defer s.scoresMutex.RUnlock()

	result := make(map[string]*ModelScore, len(s.scores))
	for k, v := range s.scores {
		scoreCopy := *v
		result[k] = &scoreCopy
	}
	return result
}

// GetBestModel returns the model with the highest score from a list of candidates
func (s *DynamicScorer) GetBestModel(candidates []string) (string, float64) {
	if len(candidates) == 0 {
		return "", 0
	}

	s.scoresMutex.RLock()
	defer s.scoresMutex.RUnlock()

	bestModel := candidates[0]
	bestScore := s.config.DefaultScore

	if score, exists := s.scores[bestModel]; exists {
		bestScore = score.CompositeScore
	}

	for _, model := range candidates[1:] {
		modelScore := s.config.DefaultScore
		if score, exists := s.scores[model]; exists {
			modelScore = score.CompositeScore
		}
		if modelScore > bestScore {
			bestScore = modelScore
			bestModel = model
		}
	}

	return bestModel, bestScore
}

// updateScores recomputes scores for all models
func (s *DynamicScorer) updateScores() {
	if s.metricsProvider == nil {
		ScoringUpdateCounter.WithLabelValues("no_provider").Inc()
		return
	}

	allMetrics, err := s.metricsProvider.GetAllModelMetrics(s.timeWindow)
	if err != nil {
		logging.Errorf("Failed to get model metrics for scoring: %v", err)
		ScoringUpdateCounter.WithLabelValues("error").Inc()
		return
	}

	s.scoresMutex.Lock()
	defer s.scoresMutex.Unlock()

	now := time.Now()
	updatedCount := 0

	for model, metrics := range allMetrics {
		score := s.computeScore(model, metrics)
		score.LastUpdated = now

		// Apply exponential moving average for smoothing
		s.emaMutex.Lock()
		if prevEMA, exists := s.emaScores[model]; exists {
			score.CompositeScore = s.config.DecayFactor*score.CompositeScore +
				(1-s.config.DecayFactor)*prevEMA
		}
		s.emaScores[model] = score.CompositeScore
		s.emaMutex.Unlock()

		s.scores[model] = score
		updatedCount++

		// Update Prometheus metrics
		DynamicScoreGauge.WithLabelValues(model).Set(score.CompositeScore)
		AccuracyScoreGauge.WithLabelValues(model).Set(score.ComponentScores.Accuracy)
		LatencyScoreGauge.WithLabelValues(model).Set(score.ComponentScores.Latency)
		CostScoreGauge.WithLabelValues(model).Set(score.ComponentScores.Cost)
	}

	ScoringUpdateCounter.WithLabelValues("success").Inc()
	logging.Debugf("Updated dynamic scores for %d models", updatedCount)
}

// computeScore computes the score for a single model
func (s *DynamicScorer) computeScore(model string, metrics ModelMetrics) *ModelScore {
	score := &ModelScore{
		Model:       model,
		Metrics:     metrics,
		SampleCount: metrics.RequestCount,
	}

	// Check if we have enough samples
	if metrics.RequestCount < s.config.MinSamples {
		score.CompositeScore = s.config.DefaultScore
		score.IsDefault = true
		score.ComponentScores = ComponentScores{
			Accuracy: s.config.DefaultScore,
			Latency:  s.config.DefaultScore,
			Cost:     s.config.DefaultScore,
		}
		return score
	}

	// Compute accuracy score (1 - error_rate)
	accuracyScore := 1.0 - metrics.ErrorRate
	if accuracyScore < 0 {
		accuracyScore = 0
	}
	if accuracyScore > 1 {
		accuracyScore = 1
	}

	// Compute latency score (normalized, lower is better)
	latencyScore := s.normalizeLatency(metrics.AvgLatencySeconds)

	// Compute cost score (normalized, lower is better)
	costScore := s.normalizeCost(metrics.CostPerKTokens)

	score.ComponentScores = ComponentScores{
		Accuracy: accuracyScore,
		Latency:  latencyScore,
		Cost:     costScore,
	}

	// Compute weighted composite score
	score.CompositeScore = s.config.Weights.Accuracy*accuracyScore +
		s.config.Weights.Latency*latencyScore +
		s.config.Weights.Cost*costScore

	// Clamp to 0-1 range
	if score.CompositeScore < 0 {
		score.CompositeScore = 0
	}
	if score.CompositeScore > 1 {
		score.CompositeScore = 1
	}

	return score
}

// normalizeLatency converts latency to a 0-1 score (lower latency = higher score)
func (s *DynamicScorer) normalizeLatency(latency float64) float64 {
	if latency <= 0 {
		return 1.0 // Perfect score for zero/negative latency
	}
	if latency >= s.config.Normalization.LatencyMax {
		return 0.0 // Worst score for max latency
	}

	// Use exponential decay: score = exp(-latency / target)
	// This gives 0.5 roughly at target and approaches 0 at max
	target := s.config.Normalization.LatencyTarget
	score := math.Exp(-latency / target)

	// Scale to ensure we hit 0 at LatencyMax
	maxScore := math.Exp(-s.config.Normalization.LatencyMax / target)
	score = (score - maxScore) / (1 - maxScore)

	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// normalizeCost converts cost to a 0-1 score (lower cost = higher score)
func (s *DynamicScorer) normalizeCost(costPerKTokens float64) float64 {
	if costPerKTokens <= 0 {
		return 1.0 // Perfect score for free
	}
	if costPerKTokens >= s.config.Normalization.CostMax {
		return 0.0 // Worst score for max cost
	}

	// Linear interpolation between 0 and max
	target := s.config.Normalization.CostTarget
	max := s.config.Normalization.CostMax

	// Score = 1 at 0, 0.5 at target, 0 at max
	if costPerKTokens <= target {
		// Linear from 1 to 0.5
		return 1.0 - 0.5*(costPerKTokens/target)
	}
	// Linear from 0.5 to 0
	return 0.5 * (1 - (costPerKTokens-target)/(max-target))
}

// ForceUpdate triggers an immediate score update
func (s *DynamicScorer) ForceUpdate() {
	s.updateScores()
}

// IsEnabled returns whether dynamic scoring is enabled
func (s *DynamicScorer) IsEnabled() bool {
	return s.config.Enabled
}

// IsRunning returns whether the scorer is currently running
func (s *DynamicScorer) IsRunning() bool {
	s.runningMutex.Lock()
	defer s.runningMutex.Unlock()
	return s.running
}

// GetConfig returns the current configuration
func (s *DynamicScorer) GetConfig() DynamicScoringConfig {
	return s.config
}
