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

// Prometheus metrics for model selection.
// See: https://github.com/vllm-project/semantic-router/issues/1093
var (
	// ModelSelectionTotal tracks the total number of model selections
	// Labels: method (static/random/multi_factor/systemone), model, decision
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

	// ModelSelectionHistory tracks selection counts per algorithm over time
	// This counter enables trend analysis of algorithm usage
	// Labels: method, decision
	ModelSelectionHistory *prometheus.CounterVec

	// ModelSelectionDependencyHealth tracks external dependency reachability
	// Labels: method, dependency, type
	ModelSelectionDependencyHealth *prometheus.GaugeVec

	metricsInitOnce sync.Once
	metricsEnabled  bool
)

// InitializeMetrics initializes the Prometheus metrics for model selection.
// This must be called during router startup to enable metrics collection.
// Metrics are registered with promauto and automatically exposed at /metrics endpoint.
func InitializeMetrics() {
	metricsInitOnce.Do(func() {
		registerSelectionMetrics()
		metricsEnabled = true
		preInitializeMetrics()
	})
}

func registerSelectionMetrics() {
	ModelSelectionTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_selection_total",
			Help: "Total number of model selections by method and selected model",
		},
		[]string{"method", "model", "decision", "tier"},
	)

	ModelSelectionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "llm_model_selection_duration_seconds",
			Help: "Duration of model selection operations in seconds",
			Buckets: []float64{
				0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
				0.25, 0.5, 1, 2.5, 5, 10, 30, 60,
			},
		},
		[]string{"method", "tier"},
	)

	ModelSelectionFallbackTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_selection_fallback_total",
			Help: "Total selector fallbacks by method, decision, and bounded reason",
		},
		[]string{"method", "decision", "reason"},
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
		[]string{"method", "tier"},
	)

	ModelSelectionHistory = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_model_selection_history",
			Help: "Selection count over time by algorithm type (for trend analysis)",
		},
		[]string{"method", "decision"},
	)

	ModelSelectionDependencyHealth = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_model_selection_dependency_health",
			Help: "External dependency reachability (1=healthy, 0=unreachable)",
		},
		[]string{"method", "dependency", "type"},
	)
}

// preInitializeMetrics initializes metrics with placeholder values so they appear in /metrics
// This is necessary because Prometheus Vec metrics only appear after WithLabelValues is called
func preInitializeMetrics() {
	// All selection methods - pre-initialize so they appear in Grafana dropdowns immediately
	methods := []SelectionMethod{MethodStatic, MethodRandom, MethodMultiFactor, MethodSystemOne}

	// Initialize selection metrics for all methods
	for _, selectionMethod := range methods {
		method := string(selectionMethod)
		ModelSelectionTotal.WithLabelValues(method, "_init", "_init", "supported")
		ModelSelectionDuration.WithLabelValues(method, "supported")
		ModelSelectionScore.WithLabelValues(method, "_init")
		ModelSelectionConfidence.WithLabelValues(method, "supported")
		ModelSelectionHistory.WithLabelValues(method, "_init")
		ModelSelectionFallbackTotal.WithLabelValues(method, "_init", "_init")
	}

	// Initialize dependency health metric
	ModelSelectionDependencyHealth.WithLabelValues("_init", "_init", "_init").Set(0)
}

// IsMetricsEnabled returns true if metrics have been initialized
func IsMetricsEnabled() bool {
	return metricsEnabled
}

// RecordSelectionWithTier records a model selection event with tier label
func RecordSelectionWithTier(method SelectionMethod, model string, decision string, tier AlgorithmTier, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	methodStr := string(method)
	tierStr := string(tier)

	ModelSelectionTotal.WithLabelValues(methodStr, model, decision, tierStr).Inc()
	ModelSelectionDuration.WithLabelValues(methodStr, tierStr).Observe(duration.Seconds())
	ModelSelectionScore.WithLabelValues(methodStr, model).Observe(score)
	ModelSelectionConfidence.WithLabelValues(methodStr, tierStr).Observe(confidence)
	ModelSelectionHistory.WithLabelValues(methodStr, decision).Inc()
}

// RecordDependencyHealth records the health status of an external dependency
func RecordDependencyHealth(method string, dependency string, depType string, healthy bool) {
	if !metricsEnabled {
		return
	}

	val := 0.0
	if healthy {
		val = 1.0
	}
	ModelSelectionDependencyHealth.WithLabelValues(method, dependency, depType).Set(val)
}
