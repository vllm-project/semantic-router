package config

import (
	"strings"
	"testing"
)

func decisionModelYAML(system, modules string) string {
	global := ""
	if system != "" {
		global = "    system:\n" + system
	}
	return builtInSignalsYAML(modules) + global
}

func mustParseDecisionModel(t *testing.T, raw string) *RouterConfig {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func moduleModels(cfg *RouterConfig) map[string]string {
	model, _, _ := cfg.ModalityClassifierModel()
	return map[string]string{
		"safety":                 cfg.SafetyModels.Safety.ModelID,
		"prompt_guard":           cfg.PromptGuard.ModelID,
		"domain_classifier":      cfg.CategoryModel.ModelID,
		"pii_classifier":         cfg.PIIModel.ModelID,
		"fact_check_classifier":  cfg.HallucinationMitigation.FactCheckModel.ModelID,
		"hallucination_detector": cfg.HallucinationMitigation.HallucinationModel.ModelID,
		"feedback_detector":      cfg.FeedbackDetector.ModelID,
		"modality_detector":      model,
	}
}

func systemLineValue(system *CanonicalSystemModels, key string) *string {
	for _, line := range systemLines {
		if line.key == key {
			return line.value(system)
		}
	}
	return nil
}

func withDecisionQuestion(raw string) string {
	return strings.Replace(raw, "  signals:\n", "  signals:\n    decision:\n      - name: needs_tools\n"+
		"        question: {type: noul, instructions: \"Does the request need a tool?\"}\n", 1)
}

func withDecisionSelector(raw, deployment string) string {
	selector := "        decision:\n          instructions: Which model should answer?\n"
	if deployment != "" {
		selector += "          deployment: " + deployment + "\n"
	}
	return strings.Replace(raw, "  decisions:\n", "  decisions:\n    - name: choose\n      priority: 200\n"+
		"      rules: {operator: AND, conditions: [{type: domain, name: math}]}\n"+
		"      modelRefs: [{model: general}, {model: other-model}]\n"+
		"      algorithm:\n        type: decision\n"+selector, 1)
}

func withOtherModel(raw string) string {
	return strings.Replace(raw, "  models:\n", "  models:\n    - name: other-model\n      backend_refs:\n"+
		"        - {name: other, endpoint: 127.0.0.1:8001, protocol: http}\n", 1)
}

func selectedDecisionYAML(artifact, system string) string {
	return decisionModelYAML("      decision_model: {deployment: selected}\n"+system, "") +
		"    deployments:\n      selected: {provider: model_runtime, artifact: " + artifact + ", device: cpu}\n"
}
func TestDecisionModelDefaultUsesDeclaredResource(t *testing.T) {
	cfg := mustParseDecisionModel(t, withDecisionQuestion(builtInSignalsYAML("")))
	name, resource, ok, err := cfg.DecisionModelDeployment()
	if err != nil || !ok || name != "primary" || resource.Artifact != "vllm-sr/Vela-2.0-0.3B" {
		t.Fatalf("default %s %+v %v", name, resource, err)
	}
	for _, consumer := range []string{"domain_classifier", "prompt_guard", "pii_classifier", "fact_check_classifier", "feedback_detector", "hallucination_detector"} {
		got, deployment, found, err := cfg.ImplicitTaskDeployment(consumer)
		if err != nil || !found || got != name || deployment != resource {
			t.Fatalf("%s duplicated resource: %s %+v %v", consumer, got, deployment, err)
		}
	}
	if cfg.DecisionQuestionDeployment(cfg.DecisionRules[0]) != name {
		t.Fatal("question did not inherit default deployment")
	}
}
func TestDecisionModelAcceptsGenericDeclaredFamilies(t *testing.T) {
	for _, artifact := range []string{"vllm-sr/Decision-1.0-Kai-0.6B", "vllm-sr/Decision-2.0-Kai-0.6B", "vllm-sr/Vela-2.0-4B", "provider/Future-Judgment"} {
		t.Run(artifact, func(t *testing.T) {
			cfg := mustParseDecisionModel(t, withOtherModel(withDecisionSelector(withDecisionQuestion(selectedDecisionYAML(artifact, "")), "")))
			name, deployment, ok, err := cfg.DecisionModelDeployment()
			if err != nil || !ok || name != "selected" || deployment.Artifact != artifact {
				t.Fatalf("selection %s %+v %v", name, deployment, err)
			}
			if cfg.DecisionQuestionDeployment(cfg.DecisionRules[0]) != "selected" || cfg.DecisionSelectorDeployment(*cfg.GetDecisionByName("choose").Algorithm.Decision) != "selected" {
				t.Fatal("wrong consumer default")
			}
			used := ModelRuntimeDeploymentsInUse(cfg)
			if used["selected"].Artifact != artifact {
				t.Fatalf("selected resource not in use: %v", used)
			}
			for key, resource := range used {
				if resource.Artifact == artifact && key != "selected" {
					t.Fatalf("duplicate resource: %s", key)
				}
			}
		})
	}
}
func TestDecisionModelPreservesExplicitSpecialist(t *testing.T) {
	cfg := mustParseDecisionModel(t, selectedDecisionYAML("vllm-sr/Decision-2.0-Kai-0.6B", "      prompt_guard: models/Vela-1.0-Encoder-307M-Guard\n"))
	name, resource, _, err := cfg.ImplicitTaskDeployment("prompt_guard")
	if err != nil || name == "selected" || resource.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Guard" {
		t.Fatalf("specialist lost: %s %+v %v", name, resource, err)
	}
	if got, _, _, _ := cfg.ImplicitTaskDeployment("domain_classifier"); got != "selected" {
		t.Fatalf("domain %s", got)
	}
}
func TestDecisionModelReferenceIsExactObject(t *testing.T) {
	for _, selector := range []string{"Vela-2.0-4B", "{deployment: missing}", "{deployment: vllm-sr/Decision-2.0-Kai-0.6B}"} {
		if _, err := ParseYAMLBytes([]byte(decisionModelYAML("      decision_model: "+selector+"\n", ""))); err == nil {
			t.Fatalf("accepted %s", selector)
		}
	}
}
func TestDecisionModelCanonicalExport(t *testing.T) {
	cfg := mustParseDecisionModel(t, selectedDecisionYAML("vllm-sr/Vela-2.0-4B", "      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback\n"))
	document := CanonicalConfigFromRouterConfig(cfg)
	system := document.Global.ModelCatalog.System
	if system.DecisionModel.Deployment != "selected" || system.FeedbackDetector != "models/Vela-1.0-Encoder-307M-Feedback" || system.DomainClassifier != "" {
		t.Fatalf("bad projection %+v", system)
	}
	if document.Global.ModelCatalog.Deployments["selected"].Artifact != "vllm-sr/Vela-2.0-4B" {
		t.Fatal("lost selected resource")
	}
}
