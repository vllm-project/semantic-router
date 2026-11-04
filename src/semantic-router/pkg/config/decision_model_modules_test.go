package config

import (
	"strings"
	"testing"
)

// remoteDecisionModelYAML reads every built-in signal with hosted-d3, a
// systemone deployment, as the decision model; system adds lines under
// global.model_catalog.system.
func remoteDecisionModelYAML(system string) string {
	return builtInSignalsYAML(`    deployments:
      hosted-d3:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: HOSTED_D3_TEST_KEY}
    system:
      decision_model: {deployment: hosted-d3}
` + system)
}

var followingConsumers = []string{
	"domain_classifier", "prompt_guard", "pii_classifier", "fact_check_classifier", "feedback_detector",
	"hallucination_detector", "modality_detector", "preference", "complexity", "safety.harmful",
}

func TestBuiltInSignalsFollowASystemOneDecisionModel(t *testing.T) {
	cfg := mustParseDecisionModel(t, remoteDecisionModelYAML(""))
	for _, consumer := range followingConsumers {
		name, deployment, ok, err := cfg.ImplicitTaskDeployment(consumer)
		if err != nil || !ok || name != "hosted-d3" || !deployment.IsSystemOne() {
			t.Fatalf("%s resolves to %q %+v %v, want the systemone decision model", consumer, name, deployment, err)
		}
	}
	if name, deployment, ok, err := cfg.DecisionModelResource(); err != nil || !ok || name != "hosted-d3" || !deployment.IsSystemOne() {
		t.Fatalf("decision model resource %q %+v %v", name, deployment, err)
	}
	hazard := *cfg
	hazard.SafetyModels.Hazard.ModelID = DefaultSystemModels().Hazard
	if name, deployment, ok, err := hazard.ImplicitTaskDeployment("safety.harmful.hazard"); err != nil || !ok || deployment.IsSystemOne() || name == "hosted-d3" {
		t.Fatalf("hazard resolves to %q %+v %v, want its own local model", name, deployment, err)
	}
	for name, deployment := range ModelRuntimeDeploymentsInUse(cfg) {
		if name == "hosted-d3" || deployment.Artifact == "vllm-sr/vela3-fast" {
			t.Fatalf("a module that follows the remote decision model runs a local deployment: %s %+v", name, deployment)
		}
	}
}

func TestAModuleThatNamesItsModelStaysLocalUnderASystemOneDecisionModel(t *testing.T) {
	line := mustParseDecisionModel(t, remoteDecisionModelYAML("      prompt_guard: models/vela3-fast\n"))
	moduleModel := mustParseDecisionModel(t, strings.Replace(remoteDecisionModelYAML(""),
		"    modules:\n", "    modules:\n      prompt_guard: {model_id: models/vela3-fast}\n", 1))
	for label, cfg := range map[string]*RouterConfig{"system line": line, "module model_id": moduleModel} {
		name, deployment, ok, err := cfg.ImplicitTaskDeployment("prompt_guard")
		if err != nil || !ok || deployment.IsSystemOne() || !deployment.IsModelRuntime() {
			t.Fatalf("%s: prompt_guard resolves to %q %+v %v, want a local deployment", label, name, deployment, err)
		}
		if _, planned := ModelRuntimeDeploymentsInUse(cfg)[name]; !planned {
			t.Fatalf("%s: prompt_guard's local deployment %q is not planned", label, name)
		}
		if name, _, _, _ := cfg.ImplicitTaskDeployment("domain_classifier"); name != "hosted-d3" {
			t.Fatalf("%s: domain_classifier resolves to %q, want the decision model", label, name)
		}
	}
}

func TestAModelRuntimeDecisionModelKeepsItsModules(t *testing.T) {
	cfg := mustParseDecisionModel(t, withDecisionQuestion(builtInSignalsYAML("")))
	for _, consumer := range followingConsumers {
		if _, deployment, ok, err := cfg.ImplicitTaskDeployment(consumer); err != nil || !ok || !deployment.IsModelRuntime() {
			t.Fatalf("%s resolves to %+v %v", consumer, deployment, err)
		}
	}
}

func TestSystemOnePresetsAreValidated(t *testing.T) {
	for _, tc := range []struct {
		name, deployment, want string
	}{
		{"model_runtime", "{provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu, presets: [pii]}", "apply only to systemone deployments"},
		{"twice", "{provider: systemone, endpoint: https://s.example.com/v1/systemone, model: m, presets: [pii, pii]}", `presets lists "pii" twice`},
		{"blank", `{provider: systemone, endpoint: https://s.example.com/v1/systemone, model: m, presets: [" pii"]}`, "presets must name ready-made questions"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(builtInSignalsYAML("    deployments:\n      extra: " + tc.deployment + "\n")))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("%v, want %q", err, tc.want)
			}
		})
	}
	cfg := mustParseDecisionModel(t, builtInSignalsYAML("    deployments:\n      extra: {provider: systemone, endpoint: https://s.example.com/v1/systemone, model: m, presets: [pii, halu]}\n"))
	if got := cfg.ModelDeployments["extra"].Presets; strings.Join(got, ",") != "pii,halu" {
		t.Fatalf("presets %v", got)
	}
}
