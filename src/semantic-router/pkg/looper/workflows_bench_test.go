package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Fixtures for the Flow/Workflows looper pure-function micro-benchmarks (no
// network), covering config resolution, static plan building, and final-prompt
// assembly. Constructing workflowStepResult relies on same-package access to its
// unexported fields — the reason these benches live in package looper.
var (
	benchWorkflowRoles = []config.WorkflowRoleConfig{
		{Name: "thinker", Models: []string{"model-a"}},
		{Name: "worker", Models: []string{"model-b"}},
		{Name: "verifier", Models: []string{"model-c"}},
	}
	benchWorkflowStaticCfg = workflowsExecutionConfig{
		Mode:  config.WorkflowModeStatic,
		Roles: benchWorkflowRoles,
		Final: config.WorkflowFinalConfig{Model: "model-a"},
	}
	benchWorkflowPlan = &workflowPlan{
		Steps: []workflowPlanStep{
			{ID: "thinker", Role: "thinker", Models: []string{"model-a"}},
			{ID: "worker", Role: "worker", Models: []string{"model-b"}},
			{ID: "verifier", Role: "verifier", Models: []string{"model-c"}},
		},
		Final: &workflowFinalStep{Model: "model-a", Prompt: "Synthesize the role outputs."},
	}
	benchWorkflowStepResults = []workflowStepResult{
		{
			step:      workflowPlanStep{ID: "thinker", Role: "thinker", Models: []string{"model-a"}},
			responses: []*ModelResponse{{Model: "model-a", Content: "Break the task into steps and list constraints."}},
		},
		{
			step:      workflowPlanStep{ID: "worker", Role: "worker", Models: []string{"model-b"}},
			responses: []*ModelResponse{{Model: "model-b", Content: "Concrete solution with reasoning.", ReasoningContent: "step 1 -> step 2"}},
		},
		{
			step:      workflowPlanStep{ID: "verifier", Role: "verifier", Models: []string{"model-c"}},
			responses: []*ModelResponse{{Model: "model-c", Content: "Checked against the request; looks correct."}},
		},
	}
)

// BenchmarkFlow_ResolveExecutionConfig measures resolving + normalizing the
// workflows config from a request carrying an algorithm override.
func BenchmarkFlow_ResolveExecutionConfig(b *testing.B) {
	req := &Request{
		Algorithm: &config.AlgorithmConfig{
			Workflows: &config.WorkflowsAlgorithmConfig{
				Roles:       benchWorkflowRoles,
				MaxSteps:    4,
				MaxParallel: 3,
			},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		resolveWorkflowsExecutionConfig(req)
	}
}

// BenchmarkFlow_BuildStaticPlan measures building the static plan from roles
// (per-role normalization + default prompt selection).
func BenchmarkFlow_BuildStaticPlan(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = buildStaticWorkflowPlan(benchWorkflowStaticCfg)
	}
}

// BenchmarkFlow_FormatStepResults measures rendering step results into the
// prompt body (strings.Builder over steps x responses).
func BenchmarkFlow_FormatStepResults(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		formatWorkflowStepResults(benchWorkflowStepResults)
	}
}

// BenchmarkFlow_BuildFinalPrompt measures assembling the final synthesis prompt.
func BenchmarkFlow_BuildFinalPrompt(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		buildWorkflowFinalPrompt(benchWorkflowPlan, "What is the capital of France?", "", benchWorkflowStepResults)
	}
}
