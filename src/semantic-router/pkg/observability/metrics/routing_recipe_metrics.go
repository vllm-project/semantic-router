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

// Streamed body arrival has its own buckets because a slow client upload can
// take far longer than the 10s ceiling of the routing stage histogram.
var streamedBodyArrival = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_streamed_body_arrival_seconds",
	Help:    "Time from the first request body chunk to end of stream in STREAMED or FULL_DUPLEX_STREAMED mode.",
	Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120},
}, []string{"recipe"})

var streamedBodyBytes = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_streamed_body_bytes",
	Help:    "Accumulated request body bytes at end of stream in STREAMED or FULL_DUPLEX_STREAMED mode.",
	Buckets: prometheus.ExponentialBuckets(1024, 2, 14), // 1 KiB to 8 MiB
}, []string{"recipe"})

var streamedBodyChunks = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_streamed_body_chunks",
	Help:    "Request body chunk count at end of stream in STREAMED or FULL_DUPLEX_STREAMED mode.",
	Buckets: []float64{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128},
}, []string{"recipe"})

var promptCompressionOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_prompt_compression_total",
	Help: "Prompt compression outcomes for the routing evaluation text.",
}, []string{"recipe", "outcome"})

// Prompt compression outcomes. The set is closed so the outcome label stays bounded.
const (
	PromptCompressionCompressed       = "compressed"
	PromptCompressionSkippedDisabled  = "skipped_disabled"
	PromptCompressionSkippedMinLength = "skipped_min_length"
	PromptCompressionSkippedMaxTokens = "skipped_max_tokens"
)

// RoutingStagePromptCompression is the llm_routing_stage_duration_seconds
// stage for prompt compression of the routing evaluation text.
const RoutingStagePromptCompression = "prompt_compression"

func ObserveStreamedBodyArrival(recipe string, seconds float64, bytes, chunks int) {
	recipe = labelOrUnknown(recipe)
	streamedBodyArrival.WithLabelValues(recipe).Observe(seconds)
	streamedBodyBytes.WithLabelValues(recipe).Observe(float64(bytes))
	streamedBodyChunks.WithLabelValues(recipe).Observe(float64(chunks))
}

func RecordPromptCompressionOutcome(recipe, outcome string) {
	promptCompressionOutcomes.WithLabelValues(labelOrUnknown(recipe), outcome).Inc()
}
