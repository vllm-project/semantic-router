package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRecipeMetricsPreserveIsolationAndBoundedLabels(t *testing.T) {
	before := testutil.ToFloat64(recipeSelections.WithLabelValues("recipe-a", "shared", "static", "model"))
	other := testutil.ToFloat64(recipeSelections.WithLabelValues("recipe-b", "shared", "static", "model"))
	RecordRecipeSelection("recipe-a", "shared", "static", "model")
	if testutil.ToFloat64(recipeSelections.WithLabelValues("recipe-a", "shared", "static", "model")) != before+1 ||
		testutil.ToFloat64(recipeSelections.WithLabelValues("recipe-b", "shared", "static", "model")) != other {
		t.Fatal("same-name decisions crossed recipe boundaries")
	}
	ObserveRoutingStage("recipe-a", "signals", 0.05)
	ObserveProjectionScore("recipe-a", "difficulty", 2.5)
	RecordEntrypointResolution("entry", "recipe-a")
	for _, name := range []string{"llm_entrypoint_requests_total", "llm_routing_stage_duration_seconds", "llm_recipe_selections_total", "llm_projection_score"} {
		if count, err := testutil.GatherAndCount(prometheus.DefaultGatherer, name); err != nil || count < 1 {
			t.Fatalf("metric %s: count=%d err=%v", name, count, err)
		}
	}
	for _, collector := range []prometheus.Collector{routingStageDuration, recipeSelections, projectionScores, entrypointRequests} {
		assertMetricHasNoLabels(t, collector, "request_id", "session_id", "trace_id", "prompt", "endpoint", "score")
	}
}

func TestStreamedBodyAndPromptCompressionMetricsUseBoundedLabels(t *testing.T) {
	ObserveStreamedBodyArrival("recipe-a", 1.5, 16384, 4)
	ObserveStreamedBodyArrival("", 0.2, 2048, 1)
	for _, outcome := range []string{
		PromptCompressionCompressed,
		PromptCompressionSkippedDisabled,
		PromptCompressionSkippedMinLength,
		PromptCompressionSkippedMaxTokens,
	} {
		RecordPromptCompressionOutcome("recipe-a", outcome)
	}
	ObserveRoutingStage("recipe-a", RoutingStagePromptCompression, 0.003)

	if got := testutil.CollectAndCount(streamedBodyArrival); got < 2 {
		t.Fatalf("passthrough arrival must fall back to the unknown recipe label, got %d series", got)
	}
	if got := testutil.ToFloat64(promptCompressionOutcomes.WithLabelValues("recipe-a", PromptCompressionCompressed)); got < 1 {
		t.Fatalf("compressed outcome not recorded: %v", got)
	}
	for _, name := range []string{
		"llm_streamed_body_arrival_seconds",
		"llm_streamed_body_bytes",
		"llm_streamed_body_chunks",
		"llm_prompt_compression_total",
	} {
		if count, err := testutil.GatherAndCount(prometheus.DefaultGatherer, name); err != nil || count < 1 {
			t.Fatalf("metric %s: count=%d err=%v", name, count, err)
		}
	}
	for _, collector := range []prometheus.Collector{streamedBodyArrival, streamedBodyBytes, streamedBodyChunks, promptCompressionOutcomes} {
		assertMetricHasNoLabels(t, collector, "request_id", "session_id", "trace_id", "prompt", "model", "endpoint")
	}
}
