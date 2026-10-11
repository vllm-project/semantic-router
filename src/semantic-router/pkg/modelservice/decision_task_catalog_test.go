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
	if response.DefaultDeployment != "primary" || len(response.Tasks) != len(BuiltinTasks()) || len(response.Models) != 22 {
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

func TestTaskCatalogDisplaysArtifactWithoutChangingDeploymentIdentity(t *testing.T) {
	for _, tc := range []struct {
		name, observedRepo, observedArtifact, declaredArtifact, want string
	}{
		{"observed model", "org/actual", "org/active", "org/declared", "org/actual"},
		{"active artifact", "", "org/active", "org/declared", "org/active"},
		{"declared artifact", "", "", "org/declared", "org/declared"},
		{"legacy served identity", "", "", "", "primary"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			cfg.ModelDeployments["primary"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: tc.declaredArtifact}
			cfg.GlobalModelBindings = map[string]config.ModelBinding{"pii_classifier": {Deployment: "primary", Contract: config.DecisionTaskContract}}
			cfg.PIIRules = []config.PIIRule{{Name: "personal", Threshold: .7}}
			cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{{Type: "pii", Name: "personal"}}}}}
			card := ModelCard{ID: "primary", Repo: tc.observedRepo, Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice", "noul", "score"}}
			status := DeploymentStatus{Name: "primary", Model: "served-alias", Artifact: tc.observedArtifact, Ready: true, Card: &card}
			response := ProjectTaskCatalog(&cfg, []DeploymentStatus{status})
			if len(response.Bindings) != 1 || response.Bindings[0].Model != tc.want || response.Bindings[0].Deployment != "primary" || !response.Bindings[0].Ready {
				t.Fatalf("binding display/identity mismatch: %+v", response.Bindings)
			}
			if len(response.Deployments) != 1 || response.Deployments[0].Model != tc.want || response.Deployments[0].Deployment != "primary" || !response.Deployments[0].Ready {
				t.Fatalf("deployment display/identity mismatch: %+v", response.Deployments)
			}
			if card.ID != "primary" || card.Repo != tc.observedRepo || response.Bindings[0].Binding.Deployment != "primary" {
				t.Fatal("display projection changed execution identity")
			}
		})
	}
}

func TestTaskCatalogDeclaredModelAndObservedModelWithoutConfig(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	response := ProjectTaskCatalog(&cfg, nil)
	if len(response.Deployments) != 1 || response.Deployments[0].Model != cfg.ModelDeployments["primary"].Artifact || response.Deployments[0].Ready {
		t.Fatalf("unloaded artifact display invented readiness: %+v", response.Deployments)
	}
	card := ModelCard{ID: "primary", Repo: "org/actual", Surfaces: []string{"decisions"}}
	response = ProjectTaskCatalog(nil, []DeploymentStatus{{Name: "primary", Model: "primary", Card: &card, Ready: true}})
	if len(response.Deployments) != 1 || response.Deployments[0].Model != "org/actual" {
		t.Fatalf("observed artifact requires authored config: %+v", response.Deployments)
	}
}
