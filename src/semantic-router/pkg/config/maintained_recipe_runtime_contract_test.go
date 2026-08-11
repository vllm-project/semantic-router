package config

import (
	"reflect"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestMaintainedRecipeExternalAliasesDoNotUseQwenReasoningFamily(t *testing.T) {
	for _, recipe := range []string{
		"accuracy",
		"agent",
		"balance",
		"feedback",
		"knowledge",
		"privacy",
	} {
		t.Run(recipe, func(t *testing.T) {
			canonical := readCanonicalRecipeConfig(t, recipe)
			for _, model := range canonical.Providers.Models {
				if !isExternalProviderModelID(model.ProviderModelID) {
					continue
				}
				if model.ReasoningFamily == "qwen3" {
					t.Fatalf(
						"external model %q (%s) must not receive Qwen reasoning parameters",
						model.Name,
						model.ProviderModelID,
					)
				}
			}
		})
	}
}

func TestAccuracyRecipeOrchestrationContract(t *testing.T) {
	data := mustReadRepoFile(t, "config/recipes/accuracy/config.yaml")
	cfg, err := ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse accuracy recipe: %v", err)
	}
	decisions := decisionsByName(cfg.Decisions)
	workflow := decisions["accuracy_workflow"]
	longContext := decisions["accuracy_long_context_direct"]
	deliberation := decisions["accuracy_deliberation"]
	direct := decisions["accuracy_direct"]
	assertAccuracyPriorityContract(t, workflow, longContext, deliberation, direct)
	assertAccuracyWorkflowContract(t, workflow)
	assertAccuracyFusionContract(t, deliberation)
}

func assertAccuracyPriorityContract(
	t *testing.T,
	workflow Decision,
	longContext Decision,
	deliberation Decision,
	direct Decision,
) {
	t.Helper()
	if workflow.Priority <= longContext.Priority ||
		longContext.Priority <= deliberation.Priority ||
		deliberation.Priority <= direct.Priority {
		t.Fatalf(
			"accuracy priority contract changed: workflow=%d long=%d deliberation=%d direct=%d",
			workflow.Priority,
			longContext.Priority,
			deliberation.Priority,
			direct.Priority,
		)
	}
}

func assertAccuracyWorkflowContract(t *testing.T, workflow Decision) {
	t.Helper()
	if workflow.Algorithm == nil || workflow.Algorithm.Workflows == nil {
		t.Fatal("accuracy_workflow must use workflows")
	}
	workflows := workflow.Algorithm.Workflows
	if workflows.Planner.Model != "qwen-coordinator" ||
		workflows.Planner.MaxCompletionTokens != 2048 ||
		workflows.MaxSteps != 4 ||
		workflows.MaxParallel != 3 ||
		workflows.MinSuccessfulResponses != 2 ||
		workflows.MaxCompletionTokens != 8192 ||
		workflows.OnError != WorkflowOnErrorSkip {
		t.Fatalf("accuracy_workflow bounds changed: %#v", workflows)
	}
}

func assertAccuracyFusionContract(t *testing.T, deliberation Decision) {
	t.Helper()
	if deliberation.Algorithm == nil || deliberation.Algorithm.Fusion == nil {
		t.Fatal("accuracy_deliberation must use fusion")
	}
	fusion := deliberation.Algorithm.Fusion
	if fusion.Model != "qwen-coordinator" ||
		fusion.MaxConcurrent != 3 ||
		fusion.MinSuccessfulResponses != 2 ||
		!reflect.DeepEqual(
			fusion.AnalysisModels,
			[]string{"opus48-worker", "gemini31-worker", "gpt55-worker"},
		) ||
		fusion.OnError != FusionOnErrorSkip {
		t.Fatalf("accuracy_deliberation degradation contract changed: %#v", fusion)
	}
	if !fusionJudgeReasoningDisabled(deliberation.ModelRefs) {
		t.Fatal("accuracy_deliberation must disable reasoning for its structured Fusion judge")
	}
}

func fusionJudgeReasoningDisabled(modelRefs []ModelRef) bool {
	for _, modelRef := range modelRefs {
		if modelRef.Model == "qwen-coordinator" &&
			modelRef.UseReasoning != nil &&
			!*modelRef.UseReasoning {
			return true
		}
	}
	return false
}

func TestAccuracyRecipeUsesCurrentOpenRouterWorkerIDs(t *testing.T) {
	canonical := readCanonicalRecipeConfig(t, "accuracy")
	want := map[string]string{
		"opus48-worker":   "anthropic/claude-opus-4.8",
		"gemini31-worker": "google/gemini-3.1-pro-preview",
		"gpt55-worker":    "openai/gpt-5.5",
	}
	for _, model := range canonical.Providers.Models {
		expected, ok := want[model.Name]
		if !ok {
			continue
		}
		if model.ProviderModelID != expected {
			t.Fatalf(
				"accuracy worker %q provider_model_id = %q, want %q",
				model.Name,
				model.ProviderModelID,
				expected,
			)
		}
		delete(want, model.Name)
	}
	if len(want) > 0 {
		t.Fatalf("accuracy recipe is missing worker aliases: %v", want)
	}
}

func readCanonicalRecipeConfig(t *testing.T, recipe string) CanonicalConfig {
	t.Helper()
	var canonical CanonicalConfig
	rel := "config/recipes/" + recipe + "/config.yaml"
	if err := yamlv3.Unmarshal(mustReadRepoFile(t, rel), &canonical); err != nil {
		t.Fatalf("decode %s: %v", rel, err)
	}
	return canonical
}

func isExternalProviderModelID(modelID string) bool {
	for _, prefix := range []string{"anthropic/", "google/", "openai/"} {
		if strings.HasPrefix(modelID, prefix) {
			return true
		}
	}
	return false
}

func decisionsByName(decisions []Decision) map[string]Decision {
	result := make(map[string]Decision, len(decisions))
	for _, decision := range decisions {
		result[decision.Name] = decision
	}
	return result
}
