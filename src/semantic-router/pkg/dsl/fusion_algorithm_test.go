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
    analysis_mode: "one_call"
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
							AnalysisMode:                 config.FusionAnalysisModeOneCall,
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
		`analysis_mode: "one_call"`,
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

func TestFusionAnalysisModeRoundTrip(t *testing.T) {
	for _, mode := range []string{
		config.FusionAnalysisModeSeparate,
		config.FusionAnalysisModeOneCall,
		config.FusionAnalysisModeNone,
	} {
		t.Run(mode, func(t *testing.T) {
			input := `
ROUTE fusion_mode {
  MODEL "judge-model", "panel-a"
  ALGORITHM fusion {
    model: "judge-model"
    analysis_models: ["panel-a"]
    analysis_mode: "` + mode + `"
  }
}`
			compiled, errs := Compile(input)
			if len(errs) > 0 {
				t.Fatalf("compile errors: %v", errs)
			}
			fusion := compiled.Decisions[0].Algorithm.Fusion
			if fusion.AnalysisMode != mode {
				t.Fatalf("compiled analysis mode = %q, want %q", fusion.AnalysisMode, mode)
			}

			decompiled, err := DecompileRouting(compiled)
			if err != nil {
				t.Fatalf("DecompileRouting error: %v", err)
			}
			if !strings.Contains(decompiled, `analysis_mode: "`+mode+`"`) {
				t.Fatalf("decompiled DSL missing analysis mode %q:\n%s", mode, decompiled)
			}
		})
	}
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
	if fusion.AnalysisMode != config.FusionAnalysisModeOneCall {
		t.Fatalf("analysis mode = %q", fusion.AnalysisMode)
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

// The quorum-failure policy is a flat scalar so it survives the DSL round trip,
// unlike the nested grounding block.
func TestFusionQuorumFailurePolicyRoundTrip(t *testing.T) {
	input := `
ROUTE fusion_quorum {
  PRIORITY 10
  MODEL "judge-model", "panel-a", "panel-b"
  ALGORITHM fusion {
    model: "judge-model"
    analysis_models: ["panel-a", "panel-b"]
    min_successful_responses: 2
    quorum_failure_policy: "fallback"
    quorum_fallback_target: "backup-model"
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	fusion := cfg.Decisions[0].Algorithm.Fusion
	if fusion.QuorumFailurePolicy != config.FusionQuorumFailurePolicyFallback {
		t.Fatalf("quorum_failure_policy = %q, want %q",
			fusion.QuorumFailurePolicy, config.FusionQuorumFailurePolicyFallback)
	}
	if fusion.QuorumFallbackTarget != "backup-model" {
		t.Fatalf("quorum_fallback_target = %q, want %q", fusion.QuorumFallbackTarget, "backup-model")
	}

	fields := map[string]Value{}
	fusionAlgorithmToFields(fusion, fields)
	policy, ok := fields["quorum_failure_policy"]
	if !ok {
		t.Fatal("decompiler dropped quorum_failure_policy")
	}
	if got := policy.(StringValue).V; got != "fallback" {
		t.Fatalf("decompiled policy = %q, want %q", got, "fallback")
	}
	target, ok := fields["quorum_fallback_target"]
	if !ok {
		t.Fatal("decompiler dropped quorum_fallback_target")
	}
	if got := target.(StringValue).V; got != "backup-model" {
		t.Fatalf("decompiled target = %q, want %q", got, "backup-model")
	}
}
