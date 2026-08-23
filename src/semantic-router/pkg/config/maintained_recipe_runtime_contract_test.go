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
			for _, model := range canonical.Models {
				for _, connection := range model.Connections {
					if !isExternalProviderModelID(connection.Model) {
						continue
					}
					if model.Card.Reasoning.Type == ReasoningFamilyTypeChatTemplateKwargs {
						t.Fatalf(
							"external model %q (%s) must not receive Qwen reasoning parameters",
							model.Name,
							connection.Model,
						)
					}
				}
			}
		})
	}
}

func TestAccuracyRecipeOrchestrationContract(t *testing.T) {
	data := mustReadRepoFile(t, "config/recipes/accuracy/config.yaml")
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse accuracy recipe: %v", err)
	}
	recipe, ok := cfg.RecipeForRequestModel("vllm-sr/auto")
	if !ok {
		t.Fatal("accuracy Entrypoint did not resolve")
	}
	decisions := decisionsByName(recipe.Profile.Decisions)
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
	assigned := modelNamesFromRefs(workflow.ModelRefs)
	if len(assigned) == 0 {
		t.Fatal("accuracy_workflow must receive Models from its Entrypoint assignment")
	}
	workflows := workflow.Algorithm.Workflows
	if workflows.Planner.Model != assigned[0] ||
		workflows.Final.Model != assigned[0] ||
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
	assigned := modelNamesFromRefs(deliberation.ModelRefs)
	if len(assigned) != 4 {
		t.Fatalf("accuracy_deliberation assignments = %v, want four Models", assigned)
	}
	fusion := deliberation.Algorithm.Fusion
	if fusion.Model != assigned[0] ||
		fusion.MaxConcurrent != 3 ||
		fusion.MinSuccessfulResponses != 2 ||
		!reflect.DeepEqual(fusion.AnalysisModels, assigned) ||
		fusion.OnError != FusionOnErrorSkip {
		t.Fatalf("accuracy_deliberation degradation contract changed: %#v", fusion)
	}
}

func TestAccuracyRecipeUsesCurrentOpenRouterWorkerIDs(t *testing.T) {
	canonical := readCanonicalRecipeConfig(t, "accuracy")
	want := map[string]string{
		"opus48-worker":   "anthropic/claude-opus-4.8",
		"gemini31-worker": "google/gemini-3.1-pro-preview",
		"gpt55-worker":    "openai/gpt-5.5",
	}
	for _, model := range canonical.Models {
		expected, ok := want[model.Name]
		if !ok {
			continue
		}
		if len(model.Connections) != 1 || model.Connections[0].Model != expected {
			t.Fatalf(
				"accuracy worker %q connections = %+v, want provider model %q",
				model.Name,
				model.Connections,
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
