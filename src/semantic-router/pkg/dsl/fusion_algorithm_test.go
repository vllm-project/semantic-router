package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileFusionAlgorithm(t *testing.T) {
	input := `
ROUTE fusion_reasoning {
  PRIORITY 10
  MODEL "judge-model", "panel-a", "panel-b"
  ALGORITHM fusion {
    model: "judge-model"
    analysis_models: ["panel-a", "panel-b"]
    max_concurrent: 2
    max_completion_tokens: 1024
    round_timeout_seconds: 90
    min_successful_responses: 1
    temperature: 0.2
    include_analysis: true
    include_intermediate_responses: true
    on_error: "skip"
    judge_prompt_version: "fusion-v1"
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	fusion := cfg.Decisions[0].Algorithm.Fusion
	assertFusionAlgorithmConfig(t, fusion)
}

func TestDecompileFusionAlgorithmRoundTrip(t *testing.T) {
	includeAnalysis := true
	includeResponses := true
	temperature := 0.2
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:     "fusion-reasoning",
					Priority: 10,
					ModelRefs: []config.ModelRef{
						{Model: "judge-model"},
						{Model: "panel-a"},
						{Model: "panel-b"},
					},
					Algorithm: &config.AlgorithmConfig{
						Type: "fusion",
						Fusion: &config.FusionAlgorithmConfig{
							Model:                        "judge-model",
							AnalysisModels:               []string{"panel-a", "panel-b"},
							MaxConcurrent:                2,
							MaxCompletionTokens:          1024,
							RoundTimeoutSeconds:          90,
							MinSuccessfulResponses:       1,
							Temperature:                  &temperature,
							IncludeAnalysis:              &includeAnalysis,
							IncludeIntermediateResponses: &includeResponses,
							OnError:                      "skip",
							JudgePromptVersion:           "fusion-v1",
						},
					},
				},
			},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	for _, want := range []string{
		"ALGORITHM fusion",
		`model: "judge-model"`,
		`analysis_models: ["panel-a", "panel-b"]`,
		"round_timeout_seconds: 90",
		"min_successful_responses: 1",
	} {
		if !strings.Contains(dslText, want) {
			t.Fatalf("decompiled DSL missing %q:\n%s", want, dslText)
		}
	}

	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v\n%s", errs, dslText)
	}
	assertFusionAlgorithmConfig(t, roundTripped.Decisions[0].Algorithm.Fusion)
}

func assertFusionAlgorithmConfig(t *testing.T, fusion *config.FusionAlgorithmConfig) {
	t.Helper()
	if fusion == nil {
		t.Fatal("expected fusion config")
	}
	assertFusionModels(t, fusion)
	assertFusionLimits(t, fusion)
	assertFusionControls(t, fusion)
	assertFusionPolicy(t, fusion)
}

func assertFusionModels(t *testing.T, fusion *config.FusionAlgorithmConfig) {
	t.Helper()
	if fusion.Model != "judge-model" {
		t.Fatalf("model = %q", fusion.Model)
	}
	if got := strings.Join(fusion.AnalysisModels, ","); got != "panel-a,panel-b" {
		t.Fatalf("analysis models = %#v", fusion.AnalysisModels)
	}
}

func assertFusionLimits(t *testing.T, fusion *config.FusionAlgorithmConfig) {
	t.Helper()
	if fusion.MaxConcurrent != 2 || fusion.MaxCompletionTokens != 1024 {
		t.Fatalf("unexpected panel limits: %#v", fusion)
	}
	if fusion.RoundTimeoutSeconds != 90 || fusion.MinSuccessfulResponses != 1 {
		t.Fatalf("unexpected quorum config: %#v", fusion)
	}
}

func assertFusionControls(t *testing.T, fusion *config.FusionAlgorithmConfig) {
	t.Helper()
	if fusion.Temperature == nil || *fusion.Temperature != 0.2 {
		t.Fatalf("temperature = %#v", fusion.Temperature)
	}
	if fusion.IncludeAnalysis == nil || !*fusion.IncludeAnalysis {
		t.Fatalf("include analysis = %#v", fusion.IncludeAnalysis)
	}
	if fusion.IncludeIntermediateResponses == nil || !*fusion.IncludeIntermediateResponses {
		t.Fatalf("include responses = %#v", fusion.IncludeIntermediateResponses)
	}
}

func assertFusionPolicy(t *testing.T, fusion *config.FusionAlgorithmConfig) {
	t.Helper()
	if fusion.OnError != "skip" || fusion.JudgePromptVersion != "fusion-v1" {
		t.Fatalf("unexpected policy fields: %#v", fusion)
	}
}
