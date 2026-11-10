package config

import (
	"reflect"
	"strings"
	"testing"
)

func decisionModelYAML(system, modules string) string {
	catalog := modules
	if system != "" {
		catalog += "    system:\n" + system
	}
	return builtInSignalsYAML(catalog)
}

func mustParseDecisionModel(t *testing.T, raw string) *RouterConfig {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func withDecisionQuestion(raw string) string {
	return strings.Replace(raw, "\n    signals:\n", "\n    signals:\n      systemone:\n        - name: needs_tools\n"+
		"          question: {type: noul, instructions: \"Does the request need a tool?\"}\n", 1)
}

func withDecisionSelector(raw, deployment string) string {
	selector := "          systemone:\n            instructions: Which model should answer?\n"
	if deployment != "" {
		selector += "            deployment: " + deployment + "\n"
	}
	return strings.Replace(raw, "\n    decisions:\n", "\n    decisions:\n      - name: choose\n        priority: 200\n"+
		"        rules: {operator: AND, conditions: [{type: domain, name: math}]}\n"+
		"        modelRefs: [{model: general}, {model: other-model}]\n"+
		"        algorithm:\n          type: systemone\n"+selector, 1)
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
	if err != nil || !ok || name != "primary" || resource.Artifact != "vllm-sr/vela3-fast" {
		t.Fatalf("default %s %+v %v", name, resource, err)
	}
	for _, consumer := range []string{"prompt_guard", "pii_classifier", "hallucination_detector"} {
		resolved, found, err := cfg.ResolveTask(consumer)
		if err != nil || !found || resolved.Explicit() || resolved.Name != name || !reflect.DeepEqual(resolved.Deployment, resource) {
			t.Fatalf("%s duplicated resource: %+v %v", consumer, resolved, err)
		}
	}
	if cfg.SystemOneRuleDeployment(cfg.SystemOneRules[0]) != name {
		t.Fatal("question did not inherit default deployment")
	}
	for _, rule := range cfg.ApplicationRules() {
		if rule.Rule.Deployment != "" && rule.Rule.Deployment != name {
			t.Fatalf("%s:%s did not inherit the default deployment: %q", rule.Type, rule.Rule.Name, rule.Rule.Deployment)
		}
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
			if cfg.SystemOneRuleDeployment(cfg.SystemOneRules[0]) != "selected" || cfg.SystemOneSelectorDeployment(*cfg.GetDecisionByName("choose").Algorithm.SystemOne) != "selected" {
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
	cfg := mustParseDecisionModel(t, selectedDecisionYAML("vllm-sr/Decision-2.0-Kai-0.6B", "")+
		"      guard: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}\n"+
		"    bindings:\n      jailbreak: {deployment: guard}\n")
	if deployment, source := cfg.ApplicationTypeDeployment(SignalTypeJailbreak); deployment != "guard" || source != "global" {
		t.Fatalf("specialist lost: %s (%s)", deployment, source)
	}
	if deployment, _ := cfg.ApplicationTypeDeployment(SignalTypePII); deployment != "selected" {
		t.Fatalf("pii %s", deployment)
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
	cfg := mustParseDecisionModel(t, selectedDecisionYAML("vllm-sr/Vela-2.0-4B", "")+
		"      pii-specialist: {provider: model_runtime, artifact: vllm-sr/Vela-2.0-0.8B, device: cpu}\n"+
		"    bindings:\n      pii: {deployment: pii-specialist}\n")
	document := CanonicalConfigFromRouterConfig(cfg)
	catalog := document.Global.ModelCatalog
	if catalog.System != (CanonicalSystemModels{DecisionModel: DecisionModelBinding{Deployment: "selected"}}) {
		t.Fatalf("the export writes only the decision model under system: %+v", catalog.System)
	}
	pii, ok := catalog.Bindings[SignalTypePII]
	if !ok || catalog.Deployments[pii.Deployment].Artifact != "vllm-sr/Vela-2.0-0.8B" {
		t.Fatalf("the pii binding is exported: %+v", catalog.Bindings)
	}
	if _, bound := catalog.Bindings[SignalTypeJailbreak]; bound {
		t.Fatal("a type that follows the decision model is not bound")
	}
	if document.Global.ModelCatalog.Deployments["selected"].Artifact != "vllm-sr/Vela-2.0-4B" {
		t.Fatal("lost selected resource")
	}
}
