package modelservice

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTaskCatalogSeparatesAvailableTasksFromActiveBindings(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	response := ProjectTaskCatalog(&cfg, nil)
	if len(response.Bindings) != 0 {
		t.Fatalf("unused task rows: %+v", response.Bindings)
	}
	if response.DefaultDeployment != "primary" || len(response.Tasks) != len(BuiltinTasks()) || len(response.Models) != 17 {
		t.Fatalf("catalog unavailable: %d %d", len(response.Tasks), len(response.Models))
	}
	if response.DefaultBindings["pii_classifier"].Contract != config.DecisionTaskContract {
		t.Fatal("generic default falsely requires token locations")
	}
	for _, model := range response.Models {
		for _, task := range model.Tasks {
			if task.TaskID == "pii_presence" && !task.Supported {
				t.Fatalf("family gate: %s", model.Model)
			}
		}
	}
	for _, deployment := range response.Deployments {
		if deployment.Ready {
			t.Fatal("declared inventory pretended ready")
		}
	}
	for _, task := range response.Tasks {
		if task.ID == "pii_spans" && (len(task.Consumers) != 1 || !task.Consumers[0].Optional || task.Consumers[0].Binding != "pii_classifier") {
			t.Fatal("span consumer disabled presence")
		}
	}
}

func TestTaskCatalogPreservesExplicitSpecialistBindingAndDemand(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"pii_classifier": {Deployment: "specialist", Contract: config.RemoteClassifierContractTokenSpans}}
	cfg.ModelDeployments["specialist"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "org/pii-specialist"}
	cfg.PIIRules = []config.PIIRule{{Name: "personal", Threshold: .7}}
	cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{{Type: "pii", Name: "personal"}}}}}
	card := ModelCard{ID: "org/pii-specialist", Surfaces: []string{"classify"}, Heads: []HeadCard{{Kind: "token", Labels: []string{"O", "PERSON"}}}}
	response := ProjectTaskCatalog(&cfg, []DeploymentStatus{{Name: "specialist", Model: card.ID, Ready: true, Card: &card}})
	if len(response.Bindings) != 1 || response.Bindings[0].Deployment != "specialist" || response.Bindings[0].Source != "global" || !response.Bindings[0].Ready || response.Bindings[0].Binding.Contract != config.RemoteClassifierContractTokenSpans {
		t.Fatalf("specialist provenance lost: %+v", response.Bindings)
	}
	if response.GlobalBindings["pii_classifier"].Deployment != "specialist" {
		t.Fatal("inactive authored defaults unavailable to editor")
	}
}
