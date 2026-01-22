/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Prometheus metrics for model selection evolution tracking
// These metrics enable explainability and traceability of selector evolution over time.
// See: https://github.com/vllm-project/semantic-router/issues/1093
var (
	// ModelSelectionTotal tracks the total number of model selections
	// Labels: method (elo/router_dc/automix/hybrid/static), model, decision
	ModelSelectionTotal *prometheus.CounterVec

	// ModelSelectionDuration tracks the duration of model selection operations
	// Labels: method
	ModelSelectionDuration *prometheus.HistogramVec

	// ModelSelectionScore tracks the score of selected models
	// Labels: method, model
	ModelSelectionScore *prometheus.HistogramVec

	// ModelSelectionConfidence tracks confidence scores of selections
	// This histogram shows the distribution of confidence scores across all selections
	// Labels: method
	ModelSelectionConfidence *prometheus.HistogramVec

	// ModelEloRating tracks current Elo ratings for models by category
	// This gauge enables monitoring rating evolution over time in Grafana
	// Labels: model, category
	ModelEloRating *prometheus.GaugeVec

	// ModelFeedbackTotal tracks feedback events (wins/losses/ties)
	// Labels: winner, loser, is_tie, category
	ModelFeedbackTotal *prometheus.CounterVec

	// ModelRatingChange tracks the distribution of rating changes during feedback updates
	// Positive values indicate rating increases, negative values indicate decreases
	// Labels: model, category, feedback_type (win/loss/tie)
	ModelRatingChange *prometheus.HistogramVec

	// ModelSelectionHistory tracks selection counts per algorithm over time
	// This counter enables trend analysis of algorithm usage
	// Labels: method, decision
	ModelSelectionHistory *prometheus.CounterVec

	// ComponentAgreement tracks how often hybrid selector components agree on selection
	// Labels: (none - global histogram)
	ComponentAgreement *prometheus.HistogramVec

	// ModelComparisons tracks the total number of comparisons a model has participated in
	// Labels: model, category
	ModelComparisons *prometheus.GaugeVec

	// ModelWinRate tracks the win rate for each model (for observability dashboards)
	// Labels: model, category
	ModelWinRate *prometheus.GaugeVec

	// --- AutoMix-specific metrics ---

	// AutoMixVerificationProb tracks the learned verification probability per model
	// This evolves as the model receives feedback (arXiv:2310.12963)
	// Labels: model
	AutoMixVerificationProb *prometheus.GaugeVec

	// AutoMixQuality tracks the learned average quality score per model
	// Labels: model
	AutoMixQuality *prometheus.GaugeVec

	// AutoMixSuccessRate tracks the query success rate per model
	// Labels: model
	AutoMixSuccessRate *prometheus.GaugeVec

	// --- RouterDC-specific metrics ---

	// RouterDCSimilarity tracks the distribution of query-model similarity scores
	// Labels: model
	RouterDCSimilarity *prometheus.HistogramVec

	// RouterDCAffinity tracks learned affinity updates from feedback
	// Labels: model
	RouterDCAffinity *prometheus.GaugeVec

	metricsInitOnce sync.Once
	metricsEnabled  bool
)

// InitializeMetrics initializes the Prometheus metrics for model selection.
// This must be called during router startup to enable metrics collection.
// Metrics are registered with promauto and automatically exposed at /metrics endpoint.
func InitializeMetrics() {
	metricsInitOnce.Do(func() {
		ModelSelectionTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_model_selection_total",
				Help: "Total number of model selections by method and selected model",
			},
			[]string{"method", "model", "decision"},
		)

		ModelSelectionDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_duration_seconds",
				Help:    "Duration of model selection operations in seconds",
				Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
			},
			[]string{"method"},
		)

		ModelSelectionScore = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_score",
				Help:    "Score of selected models (normalized 0-1)",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"method", "model"},
		)

		ModelSelectionConfidence = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_confidence",
				Help:    "Confidence score distribution of model selections",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"method"},
		)

		ModelEloRating = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_elo_rating",
				Help: "Current Elo rating for models by category (enables evolution tracking)",
			},
			[]string{"model", "category"},
		)

		ModelFeedbackTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_model_feedback_total",
				Help: "Total feedback events (wins/losses/ties) by model pair and category",
			},
			[]string{"winner", "loser", "is_tie", "category"},
		)

		ModelRatingChange = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "llm_model_rating_change",
				Help: "Distribution of Elo rating changes during feedback updates",
				// Buckets cover typical Elo K-factor based changes (-32 to +32)
				Buckets: []float64{-32, -24, -16, -8, -4, -2, 0, 2, 4, 8, 16, 24, 32},
			},
			[]string{"model", "category", "feedback_type"},
		)

		ModelSelectionHistory = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_model_selection_history",
				Help: "Selection count over time by algorithm type (for trend analysis)",
			},
			[]string{"method", "decision"},
		)

		ComponentAgreement = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_component_agreement",
				Help:    "Agreement ratio between hybrid selector components (1.0 = all agree)",
				Buckets: []float64{0.0, 0.25, 0.5, 0.75, 1.0},
			},
			[]string{},
		)

		ModelComparisons = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_comparisons_total",
				Help: "Total number of comparisons a model has participated in",
			},
			[]string{"model", "category"},
		)

		ModelWinRate = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_win_rate",
				Help: "Win rate for each model (wins / total comparisons)",
			},
			[]string{"model", "category"},
		)

		// --- AutoMix-specific metrics ---
		AutoMixVerificationProb = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_automix_verification_prob",
				Help: "Learned verification probability per model (evolves with feedback)",
			},
			[]string{"model"},
		)

		AutoMixQuality = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_automix_quality",
				Help: "Learned average quality score per model (evolves with feedback)",
			},
			[]string{"model"},
		)

		AutoMixSuccessRate = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_automix_success_rate",
				Help: "Query success rate per model (success_count / total_count)",
			},
			[]string{"model"},
		)

		// --- RouterDC-specific metrics ---
		RouterDCSimilarity = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_routerdc_similarity",
				Help:    "Distribution of query-model similarity scores",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"model"},
		)

		RouterDCAffinity = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_routerdc_affinity",
				Help: "Learned query-model affinity from feedback (evolves with feedback)",
			},
			[]string{"model"},
		)

		metricsEnabled = true

		// Pre-initialize metrics with placeholder labels so they appear in /metrics
		// Prometheus Vec metrics only appear after WithLabelValues is called
		preInitializeMetrics()
	})
}

// preInitializeMetrics initializes metrics with placeholder values so they appear in /metrics
// This is necessary because Prometheus Vec metrics only appear after WithLabelValues is called
// preInitializeMetrics initializes metrics with placeholder values so they appear in /metrics
// This is necessary because Prometheus Vec metrics only appear after WithLabelValues is called
func preInitializeMetrics() {
	// All selection methods - pre-initialize so they appear in Grafana dropdowns immediately
	methods := []string{"elo", "router_dc", "automix", "hybrid", "static"}

	// Initialize selection metrics for all methods
	for _, method := range methods {
		ModelSelectionTotal.WithLabelValues(method, "_init", "_init")
		ModelSelectionDuration.WithLabelValues(method)
		ModelSelectionScore.WithLabelValues(method, "_init")
		ModelSelectionConfidence.WithLabelValues(method)
		ModelSelectionHistory.WithLabelValues(method, "_init")
	}

	// Initialize Elo metrics
	ModelEloRating.WithLabelValues("_init", "_init").Set(0)
	ModelFeedbackTotal.WithLabelValues("_init", "_init", "false", "_init")
	ModelRatingChange.WithLabelValues("_init", "_init", "_init")

	// Initialize other metrics
	ComponentAgreement.WithLabelValues()
	ModelComparisons.WithLabelValues("_init", "_init").Set(0)
	ModelWinRate.WithLabelValues("_init", "_init").Set(0)

	// Initialize AutoMix metrics
	AutoMixVerificationProb.WithLabelValues("_init").Set(0)
	AutoMixQuality.WithLabelValues("_init").Set(0)
	AutoMixSuccessRate.WithLabelValues("_init").Set(0)

	// Initialize RouterDC metrics
	RouterDCSimilarity.WithLabelValues("_init")
	RouterDCAffinity.WithLabelValues("_init").Set(0)
}

// IsMetricsEnabled returns true if metrics have been initialized
func IsMetricsEnabled() bool {
	return metricsEnabled
}

// RecordSelection records a basic model selection event
func RecordSelection(method string, decision string, model string, score float64) {
	if !metricsEnabled {
		return
	}

	ModelSelectionTotal.WithLabelValues(method, model, decision).Inc()
	ModelSelectionScore.WithLabelValues(method, model).Observe(score)
	ModelSelectionHistory.WithLabelValues(method, decision).Inc()
}

// RecordSelectionFull records a model selection event with all metrics
func RecordSelectionFull(method SelectionMethod, model string, decision string, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	methodStr := string(method)

	ModelSelectionTotal.WithLabelValues(methodStr, model, decision).Inc()
	ModelSelectionDuration.WithLabelValues(methodStr).Observe(duration.Seconds())
	ModelSelectionScore.WithLabelValues(methodStr, model).Observe(score)
	ModelSelectionConfidence.WithLabelValues(methodStr).Observe(confidence)
	ModelSelectionHistory.WithLabelValues(methodStr, decision).Inc()
}

// RecordEloRating records the current Elo rating for a model in a category
func RecordEloRating(model, category string, rating float64) {
	if !metricsEnabled {
		return
	}
	ModelEloRating.WithLabelValues(model, category).Set(rating)
}

// RecordEloRatings records Elo ratings for multiple models at once
// This is useful for batch updates when loading from storage or after initialization
func RecordEloRatings(ratings map[string]*ModelRating, category string) {
	if !metricsEnabled {
		return
	}
	for _, r := range ratings {
		ModelEloRating.WithLabelValues(r.Model, category).Set(r.Rating)
		ModelComparisons.WithLabelValues(r.Model, category).Set(float64(r.Comparisons))
		if r.Comparisons > 0 {
			winRate := float64(r.Wins) / float64(r.Comparisons)
			ModelWinRate.WithLabelValues(r.Model, category).Set(winRate)
		}
	}
}

// RecordFeedback records a feedback event with category context
func RecordFeedback(winner, loser string, isTie bool, category string) {
	if !metricsEnabled {
		return
	}

	tieStr := "false"
	if isTie {
		tieStr = "true"
	}

	if loser == "" {
		loser = "none"
	}
	if category == "" {
		category = "_global"
	}

	ModelFeedbackTotal.WithLabelValues(winner, loser, tieStr, category).Inc()
}

// RecordRatingChange records an Elo rating change after feedback
func RecordRatingChange(model, category string, oldRating, newRating float64, feedbackType string) {
	if !metricsEnabled {
		return
	}

	if category == "" {
		category = "_global"
	}

	change := newRating - oldRating
	ModelRatingChange.WithLabelValues(model, category, feedbackType).Observe(change)

	// Also update the current rating gauge
	ModelEloRating.WithLabelValues(model, category).Set(newRating)
}

// RecordModelStats records model statistics (comparisons, win rate)
func RecordModelStats(model, category string, comparisons, wins, losses, ties int) {
	if !metricsEnabled {
		return
	}

	if category == "" {
		category = "_global"
	}

	ModelComparisons.WithLabelValues(model, category).Set(float64(comparisons))

	if comparisons > 0 {
		winRate := float64(wins) / float64(comparisons)
		ModelWinRate.WithLabelValues(model, category).Set(winRate)
	}
}

// RecordComponentAgreement records the agreement ratio between hybrid selector components
func RecordComponentAgreement(agreementRatio float64) {
	if !metricsEnabled {
		return
	}
	ComponentAgreement.WithLabelValues().Observe(agreementRatio)
}

// FeedbackMetrics contains pre-computed metrics for a feedback update
type FeedbackMetrics struct {
	Winner       string
	Loser        string
	Category     string
	IsTie        bool
	WinnerOldElo float64
	WinnerNewElo float64
	LoserOldElo  float64
	LoserNewElo  float64
	WinnerStats  ModelRating
	LoserStats   ModelRating
}

// RecordFeedbackMetrics records all metrics for a feedback event in one call
// This is the preferred method for recording feedback as it ensures consistency
func RecordFeedbackMetrics(m *FeedbackMetrics) {
	if !metricsEnabled || m == nil {
		return
	}

	category := m.Category
	if category == "" {
		category = "_global"
	}

	// Record feedback event
	tieStr := "false"
	if m.IsTie {
		tieStr = "true"
	}
	loser := m.Loser
	if loser == "" {
		loser = "none"
	}
	ModelFeedbackTotal.WithLabelValues(m.Winner, loser, tieStr, category).Inc()

	// Record rating changes
	if m.Winner != "" {
		feedbackType := "win"
		if m.IsTie {
			feedbackType = "tie"
		}
		change := m.WinnerNewElo - m.WinnerOldElo
		ModelRatingChange.WithLabelValues(m.Winner, category, feedbackType).Observe(change)
		ModelEloRating.WithLabelValues(m.Winner, category).Set(m.WinnerNewElo)
		ModelComparisons.WithLabelValues(m.Winner, category).Set(float64(m.WinnerStats.Comparisons))
		if m.WinnerStats.Comparisons > 0 {
			winRate := float64(m.WinnerStats.Wins) / float64(m.WinnerStats.Comparisons)
			ModelWinRate.WithLabelValues(m.Winner, category).Set(winRate)
		}
	}

	if m.Loser != "" {
		feedbackType := "loss"
		if m.IsTie {
			feedbackType = "tie"
		}
		change := m.LoserNewElo - m.LoserOldElo
		ModelRatingChange.WithLabelValues(m.Loser, category, feedbackType).Observe(change)
		ModelEloRating.WithLabelValues(m.Loser, category).Set(m.LoserNewElo)
		ModelComparisons.WithLabelValues(m.Loser, category).Set(float64(m.LoserStats.Comparisons))
		if m.LoserStats.Comparisons > 0 {
			winRate := float64(m.LoserStats.Wins) / float64(m.LoserStats.Comparisons)
			ModelWinRate.WithLabelValues(m.Loser, category).Set(winRate)
		}
	}
}

// calculateAgreementRatio calculates how many components chose the same model
func calculateAgreementRatio(choices []string) float64 {
	if len(choices) <= 1 {
		return 1.0
	}

	// Count occurrences of each choice
	counts := make(map[string]int)
	for _, c := range choices {
		counts[c]++
	}

	// Find the most common choice
	maxCount := 0
	for _, count := range counts {
		if count > maxCount {
			maxCount = count
		}
	}

	// Agreement ratio = (max agreement) / (total components)
	return float64(maxCount) / float64(len(choices))
}

// RecordHybridSelection records metrics for a hybrid selection including component agreement
func RecordHybridSelection(selectedModel string, decision string, componentChoices map[string]string, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	// Record standard selection metrics
	RecordSelectionFull(MethodHybrid, selectedModel, decision, score, confidence, duration)

	// Calculate and record component agreement
	if len(componentChoices) > 1 {
		choices := make([]string, 0, len(componentChoices))
		for _, model := range componentChoices {
			choices = append(choices, model)
		}
		agreement := calculateAgreementRatio(choices)
		RecordComponentAgreement(agreement)
	}
}

// NormalizeRatingChange normalizes a rating change to a 0-1 scale for visualization
// This is useful when comparing changes across different K-factor configurations
func NormalizeRatingChange(change, kFactor float64) float64 {
	if kFactor <= 0 {
		kFactor = 32.0 // Default K-factor
	}
	// Normalize to [-1, 1] range based on maximum possible change
	normalized := change / kFactor
	// Clamp to valid range
	return math.Max(-1.0, math.Min(1.0, normalized))
}

// --- AutoMix metrics recording functions ---

// RecordAutoMixCapability records AutoMix capability metrics for a model
func RecordAutoMixCapability(model string, verificationProb, quality float64, successCount, totalCount int) {
	if !metricsEnabled {
		return
	}

	AutoMixVerificationProb.WithLabelValues(model).Set(verificationProb)
	AutoMixQuality.WithLabelValues(model).Set(quality)

	if totalCount > 0 {
		successRate := float64(successCount) / float64(totalCount)
		AutoMixSuccessRate.WithLabelValues(model).Set(successRate)
	}
}

// --- RouterDC metrics recording functions ---

// RecordRouterDCSimilarity records a query-model similarity score
func RecordRouterDCSimilarity(model string, similarity float64) {
	if !metricsEnabled {
		return
	}
	RouterDCSimilarity.WithLabelValues(model).Observe(similarity)
}

// RecordRouterDCAffinity records the current affinity for a model
func RecordRouterDCAffinity(model string, affinity float64) {
	if !metricsEnabled {
		return
	}
	RouterDCAffinity.WithLabelValues(model).Set(affinity)
}
