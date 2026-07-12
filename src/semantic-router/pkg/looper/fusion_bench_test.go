package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Fixtures for the Fusion looper pure-function micro-benchmarks (no network),
// covering config resolution, prompt assembly, and analysis parsing.
var (
	benchFusionRefs = []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
	}
	benchFusionPanel = []*ModelResponse{
		{Model: "model-a", Content: "The capital of France is Paris."},
		{Model: "model-b", Content: "France's capital is Paris.", ReasoningContent: "France -> capital -> Paris"},
		{Model: "model-c", Content: "It is Paris."},
	}
	benchFusionAnalysis = &FusionAnalysis{
		Consensus:      []string{"The capital of France is Paris"},
		UniqueInsights: []string{"model-b supplied reasoning"},
	}
	// benchFusionAnalysisJSON is a judge response the parser must extract + unmarshal.
	benchFusionAnalysisJSON = `{"consensus":["Paris is the capital"],"contradictions":[],"unique_insights":["reasoning provided"]}`
)

// BenchmarkFusion_ResolveExecutionConfig measures the full config-resolution
// pipeline (merge algorithm/request config + normalize + grounding defaults).
func BenchmarkFusion_ResolveExecutionConfig(b *testing.B) {
	looper := NewFusionLooper(&config.LooperConfig{})
	req := &Request{ModelRefs: benchFusionRefs}
	b.ReportAllocs()
	for b.Loop() {
		looper.resolveFusionExecutionConfig(req)
	}
}

// BenchmarkFusion_FormatPanelResponses measures rendering the panel into the
// prompt body (strings.Builder over N responses).
func BenchmarkFusion_FormatPanelResponses(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		formatPanelResponses(benchFusionPanel)
	}
}

// BenchmarkFusion_BuildAnalysisPrompt measures assembling the judge analysis prompt.
func BenchmarkFusion_BuildAnalysisPrompt(b *testing.B) {
	cfg := fusionExecutionConfig{}
	b.ReportAllocs()
	for b.Loop() {
		buildFusionAnalysisPrompt(cfg, "What is the capital of France?", benchFusionPanel)
	}
}

// BenchmarkFusion_BuildFinalPrompt measures assembling the final synthesis prompt,
// including json.MarshalIndent of the structured analysis.
func BenchmarkFusion_BuildFinalPrompt(b *testing.B) {
	cfg := fusionExecutionConfig{Model: "judge"}
	b.ReportAllocs()
	for b.Loop() {
		buildFusionFinalPrompt(cfg, "What is the capital of France?", "", benchFusionPanel, benchFusionAnalysis)
	}
}

// BenchmarkFusion_ParseAnalysis measures extracting + unmarshalling the judge's
// JSON analysis response.
func BenchmarkFusion_ParseAnalysis(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = parseFusionAnalysis(benchFusionAnalysisJSON)
	}
}
