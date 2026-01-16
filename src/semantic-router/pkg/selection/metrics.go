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
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Prometheus metrics for model selection tracking
var (
	// ModelSelectionTotal tracks the total number of model selections
	ModelSelectionTotal *prometheus.CounterVec

	// ModelSelectionDuration tracks the duration of model selection
	ModelSelectionDuration *prometheus.HistogramVec

	// ModelSelectionScore tracks the score of selected models
	ModelSelectionScore *prometheus.HistogramVec

	// ModelSelectionConfidence tracks confidence of selections
	ModelSelectionConfidence *prometheus.HistogramVec

	// ModelEloRating tracks current Elo ratings for models
	ModelEloRating *prometheus.GaugeVec

	// ModelFeedbackTotal tracks feedback events
	ModelFeedbackTotal *prometheus.CounterVec

	// ComponentAgreement tracks how often components agree on selection
	ComponentAgreement *prometheus.HistogramVec

	metricsInitOnce sync.Once
)

// InitializeMetrics initializes the Prometheus metrics for model selection
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
				Help:    "Duration of model selection in seconds",
				Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
			},
			[]string{"method"},
		)

		ModelSelectionScore = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_score",
				Help:    "Score of selected models",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"method", "model"},
		)

		ModelSelectionConfidence = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_confidence",
				Help:    "Confidence of model selections",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"method"},
		)

		ModelEloRating = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_elo_rating",
				Help: "Current Elo rating for models by category",
			},
			[]string{"model", "category"},
		)

		ModelFeedbackTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_model_feedback_total",
				Help: "Total feedback events by type",
			},
			[]string{"winner", "loser", "is_tie"},
		)

		ComponentAgreement = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "llm_model_selection_component_agreement",
				Help:    "Agreement ratio between selection components (for hybrid)",
				Buckets: []float64{0.0, 0.25, 0.5, 0.75, 1.0},
			},
			[]string{},
		)
	})
}

// RecordSelection records a model selection event with full metrics
func RecordSelection(method string, decision string, model string, score float64) {
	if ModelSelectionTotal == nil {
		return // Metrics not initialized
	}

	ModelSelectionTotal.WithLabelValues(method, model, decision).Inc()
	ModelSelectionScore.WithLabelValues(method, model).Observe(score)
}

// RecordSelectionFull records a model selection event with all metrics
func RecordSelectionFull(method SelectionMethod, model string, decision string, score, confidence float64, duration time.Duration) {
	if ModelSelectionTotal == nil {
		return // Metrics not initialized
	}

	methodStr := string(method)

	ModelSelectionTotal.WithLabelValues(methodStr, model, decision).Inc()
	ModelSelectionDuration.WithLabelValues(methodStr).Observe(duration.Seconds())
	ModelSelectionScore.WithLabelValues(methodStr, model).Observe(score)
	ModelSelectionConfidence.WithLabelValues(methodStr).Observe(confidence)
}

// RecordEloRating records the current Elo rating for a model
func RecordEloRating(model, category string, rating float64) {
	if ModelEloRating == nil {
		return
	}
	ModelEloRating.WithLabelValues(model, category).Set(rating)
}

// RecordFeedback records a feedback event
func RecordFeedback(winner, loser string, isTie bool) {
	if ModelFeedbackTotal == nil {
		return
	}

	tieStr := "false"
	if isTie {
		tieStr = "true"
	}

	if loser == "" {
		loser = "none"
	}

	ModelFeedbackTotal.WithLabelValues(winner, loser, tieStr).Inc()
}

// RecordComponentAgreement records the agreement ratio between components
func RecordComponentAgreement(agreementRatio float64) {
	if ComponentAgreement == nil {
		return
	}
	ComponentAgreement.WithLabelValues().Observe(agreementRatio)
}
