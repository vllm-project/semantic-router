package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// All labels below are resolved configuration names, never caller content,
// URLs, request/session identifiers, or projection values.
var routingStageDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_routing_stage_duration_seconds",
	Help:    "Recipe routing stage duration in seconds; signals includes projection evaluation.",
	Buckets: prometheus.DefBuckets,
}, []string{"recipe", "stage"})

var recipeSelections = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_recipe_selections_total",
	Help: "Successful model selections by recipe, decision, selection method and logical model; not completed requests.",
}, []string{"recipe", "decision", "algorithm", "model"})

var entrypointRequests = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_entrypoint_requests_total",
	Help: "Requests resolved to a configured entrypoint and recipe, before selection or generation.",
}, []string{"entrypoint", "recipe"})

var projectionScores = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_projection_score",
	Help:    "Evaluated recipe projection scores, which are not necessarily probabilities.",
	Buckets: []float64{-10, -1, 0, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 100},
}, []string{"recipe", "projection"})

func ObserveRoutingStage(recipe, stage string, seconds float64) {
	routingStageDuration.WithLabelValues(labelOrUnknown(recipe), stage).Observe(seconds)
}

func RecordEntrypointResolution(entrypoint, recipe string) {
	entrypointRequests.WithLabelValues(entrypoint, recipe).Inc()
}

func RecordRecipeSelection(recipe, decision, algorithm, model string) {
	recipeSelections.WithLabelValues(recipe, labelOrUnknown(decision), labelOrUnknown(algorithm), model).Inc()
}

func ObserveProjectionScore(recipe, name string, value float64) {
	projectionScores.WithLabelValues(recipe, name).Observe(value)
}
