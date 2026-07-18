package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileWorkflowsDynamicAlgorithm(t *testing.T) {
	input := `
SIGNAL domain code { description: "coding task" }

ROUTE flow_code {
  PRIORITY 10
  WHEN domain("code")
  MODEL "qwen-coordinator", "deepseek-worker", "claude-worker"
  ALGORITHM workflows {
    mode: "dynamic"
    template: "micro_agent"
    planner: { model: "qwen-coordinator" }
    max_steps: 6
    max_parallel: 3
    max_completion_tokens: 1024
    round_timeout_seconds: 90
    min_successful_responses: 2
    temperature: 0.2
    include_intermediate_responses: true
    on_error: "fail"
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 1 {
		t.Fatalf("decisions = %d", len(cfg.Decisions))
	}
	workflows := cfg.Decisions[0].Algorithm.Workflows
	assertWorkflowsDynamicConfig(t, workflows)
}

func TestCompileWorkflowsDynamicPlannerDottedField(t *testing.T) {
	input := `
ROUTE flow_code {
  PRIORITY 10
  MODEL "qwen-coordinator", "deepseek-worker"
  ALGORITHM workflows {
    mode: "dynamic"
    planner.model: "qwen-coordinator"
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	workflows := cfg.Decisions[0].Algorithm.Workflows
	if workflows == nil || workflows.Planner.Model != "qwen-coordinator" {
		t.Fatalf("workflows planner = %#v", workflows)
	}
}

func TestCompileWorkflowsStaticRolesAlgorithm(t *testing.T) {
	input := `
ROUTE flow_static {
  PRIORITY 10
  MODEL "thinker-model", "worker-model", "verifier-model"
  ALGORITHM workflows {
    mode: "static"
    roles: [
      { name: "thinker", models: ["thinker-model"], prompt: "plan" },
      { name: "worker", models: ["worker-model"], prompt: "solve" },
      { name: "verifier", models: ["verifier-model"], prompt: "check" }
    ]
    final: { model: "verifier-model", prompt: "merge" }
    max_steps: 3
    max_parallel: 1
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	workflows := cfg.Decisions[0].Algorithm.Workflows
	if workflows == nil || len(workflows.Roles) != 3 {
		t.Fatalf("workflows roles = %#v", workflows)
	}
	if workflows.Roles[0].Name != "thinker" || workflows.Roles[0].Models[0] != "thinker-model" {
		t.Fatalf("unexpected first role: %#v", workflows.Roles[0])
	}
	if workflows.Final.Model != "verifier-model" || workflows.Final.Prompt != "merge" {
		t.Fatalf("unexpected final config: %#v", workflows.Final)
	}
}

func TestDecompileWorkflowsDynamicAlgorithmRoundTrip(t *testing.T) {
	includeTrace := true
	temperature := 0.2
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:     "flow-code",
					Priority: 10,
					ModelRefs: []config.ModelRef{
						{Model: "qwen-coordinator"},
						{Model: "deepseek-worker"},
						{Model: "claude-worker"},
					},
					Algorithm: &config.AlgorithmConfig{
						Type: "workflows",
						Workflows: &config.WorkflowsAlgorithmConfig{
							Mode:                         "dynamic",
							Template:                     "micro_agent",
							Planner:                      config.WorkflowPlannerConfig{Model: "qwen-coordinator"},
							MaxSteps:                     6,
							MaxParallel:                  3,
							MaxCompletionTokens:          1024,
							RoundTimeoutSeconds:          90,
							MinSuccessfulResponses:       2,
							Temperature:                  &temperature,
							IncludeIntermediateResponses: &includeTrace,
							OnError:                      "fail",
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
		"ALGORITHM workflows",
		`mode: "dynamic"`,
		`planner: { model: "qwen-coordinator" }`,
		"max_steps: 6",
		"max_parallel: 3",
		"round_timeout_seconds: 90",
		"min_successful_responses: 2",
	} {
		if !strings.Contains(dslText, want) {
			t.Fatalf("decompiled DSL missing %q:\n%s", want, dslText)
		}
	}

	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v\n%s", errs, dslText)
	}
	assertWorkflowsDynamicConfig(t, roundTripped.Decisions[0].Algorithm.Workflows)
}

func TestDecompileWorkflowsStaticRolesAlgorithmRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:     "flow-static",
					Priority: 10,
					ModelRefs: []config.ModelRef{
						{Model: "thinker-model"},
						{Model: "worker-model"},
						{Model: "verifier-model"},
					},
					Algorithm: &config.AlgorithmConfig{
						Type: "workflows",
						Workflows: &config.WorkflowsAlgorithmConfig{
							Mode: "static",
							Roles: []config.WorkflowRoleConfig{
								{Name: "thinker", Models: []string{"thinker-model"}, Prompt: "plan"},
								{Name: "worker", Models: []string{"worker-model"}, Prompt: "solve"},
								{Name: "verifier", Models: []string{"verifier-model"}, Prompt: "check"},
							},
							Final:       config.WorkflowFinalConfig{Model: "verifier-model", Prompt: "merge"},
							MaxSteps:    3,
							MaxParallel: 1,
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
		"roles:",
		`name: "thinker"`,
		`models: ["thinker-model"]`,
		`final: { model: "verifier-model", prompt: "merge" }`,
	} {
		if !strings.Contains(dslText, want) {
			t.Fatalf("decompiled DSL missing %q:\n%s", want, dslText)
		}
	}
	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v\n%s", errs, dslText)
	}
	roles := roundTripped.Decisions[0].Algorithm.Workflows.Roles
	if len(roles) != 3 || roles[2].Name != "verifier" {
		t.Fatalf("round-trip roles = %#v", roles)
	}
}

func assertWorkflowsDynamicConfig(t *testing.T, workflows *config.WorkflowsAlgorithmConfig) {
	t.Helper()
	if workflows == nil {
		t.Fatal("expected workflows config")
	}
	assertWorkflowsDynamicIdentity(t, workflows)
	assertWorkflowsDynamicLimits(t, workflows)
}

func assertWorkflowsDynamicIdentity(t *testing.T, workflows *config.WorkflowsAlgorithmConfig) {
	t.Helper()
	if workflows.Mode != "dynamic" {
		t.Fatalf("mode = %q", workflows.Mode)
	}
	if workflows.Template != "micro_agent" {
		t.Fatalf("template = %q", workflows.Template)
	}
	if workflows.Planner.Model != "qwen-coordinator" {
		t.Fatalf("planner model = %q", workflows.Planner.Model)
	}
}

func assertWorkflowsDynamicLimits(t *testing.T, workflows *config.WorkflowsAlgorithmConfig) {
	t.Helper()
	if workflows.MaxSteps != 6 || workflows.MaxParallel != 3 || workflows.MaxCompletionTokens != 1024 {
		t.Fatalf("limits = steps:%d parallel:%d tokens:%d", workflows.MaxSteps, workflows.MaxParallel, workflows.MaxCompletionTokens)
	}
	if workflows.RoundTimeoutSeconds != 90 || workflows.MinSuccessfulResponses != 2 {
		t.Fatalf("quorum controls = timeout:%d min_success:%d", workflows.RoundTimeoutSeconds, workflows.MinSuccessfulResponses)
	}
	if workflows.Temperature == nil || *workflows.Temperature != 0.2 {
		t.Fatalf("temperature = %#v", workflows.Temperature)
	}
	if workflows.IncludeIntermediateResponses == nil || !*workflows.IncludeIntermediateResponses {
		t.Fatalf("include_intermediate_responses = %#v", workflows.IncludeIntermediateResponses)
	}
	if workflows.OnError != "fail" {
		t.Fatalf("on_error = %q", workflows.OnError)
	}
}
