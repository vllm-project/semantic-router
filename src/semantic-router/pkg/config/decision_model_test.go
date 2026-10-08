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

func TestDecisionModelDefaultsToVela2_03B(t *testing.T) {
	cfg := mustParseDecisionModel(t, builtInSignalsYAML(""))
	if cfg.DecisionModel != DecisionModelVela2_03B {
		t.Fatalf("decision model = %q, want %s", cfg.DecisionModel, DecisionModelVela2_03B)
	}
	for module, model := range moduleModels(cfg) {
		if model != Vela2SignalModel {
			t.Fatalf("%s runs %q, want %q", module, model, Vela2SignalModel)
		}
	}
}

func TestDecisionModelBindsEveryBuiltInSignalToOneDeployment(t *testing.T) {
	sizes := map[string]struct{ model, deployment string }{
		"Vela-2.0-0.8B": {"vllm-sr/Vela-2.0-0.8B cpu exact", "@Vela-2.0-0.8B"},
		"vela-2.0-4b":   {"vllm-sr/Vela-2.0-4B auto exact", "@Vela-2.0-4B/auto"},
		"VELA-2.0-9B":   {"vllm-sr/Vela-2.0-9B auto exact", "@Vela-2.0-9B/auto"},
	}
	for name, want := range sizes {
		raw := decisionModelYAML("      decision_model: "+name+"\n", "")
		got := deploymentsByArtifact(t, raw)
		if len(got) != 1 || len(got[want.model]) != 1 || got[want.model][0] != want.deployment {
			t.Fatalf("%s: every built-in signal must run on %s (%s), got %v", name, want.deployment, want.model, got)
		}
		cfg := mustParseDecisionModel(t, raw)
		if !strings.EqualFold(cfg.DecisionModel, name) || cfg.DecisionModel == name && name != "Vela-2.0-0.8B" {
			t.Fatalf("%s: decision model resolved to %q, want its canonical spelling", name, cfg.DecisionModel)
		}
	}
}

func TestDecisionModelVela1RestoresTheSpecialists(t *testing.T) {
	cfg := mustParseDecisionModel(t, decisionModelYAML("      decision_model: Vela-1.0\n", ""))
	want := Vela1SystemModels()
	for module, model := range moduleModels(cfg) {
		expected := vela1ModalityModel
		if module != "modality_detector" {
			expected = *systemLineValue(&want, module)
		}
		if model != expected {
			t.Fatalf("%s runs %q, want %q", module, model, expected)
		}
	}
	if cfg.PromptGuard.Threshold != 0.5 || cfg.CategoryModel.Threshold != 0.5 || cfg.PIIModel.Threshold != 0.9 ||
		cfg.HallucinationMitigation.FactCheckModel.Threshold != 0.95 || cfg.FeedbackDetector.Threshold != 0.7 {
		t.Fatalf("the specialists must keep their own module thresholds, got guard %v domain %v pii %v fact check %v feedback %v",
			cfg.PromptGuard.Threshold, cfg.CategoryModel.Threshold, cfg.PIIModel.Threshold,
			cfg.HallucinationMitigation.FactCheckModel.Threshold, cfg.FeedbackDetector.Threshold)
	}
	if _, _, ok, _ := cfg.DecisionModelDeployment(); ok {
		t.Fatal("Vela 1.0 has no shared deployment for decision questions")
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

func TestDecisionModelRejectsOtherModels(t *testing.T) {
	for name, want := range map[string]string{
		"Vela-2.0-27B":                "is not a decision model; choose Vela-2.0-0.3B (the default), Vela-2.0-0.8B, Vela-2.0-4B, Vela-2.0-9B or Vela-1.0",
		"vllm-sr/Vela-2.0-4B":         "name the model Vela-2.0-4B, without a repository or path",
		"models/Vela-2.0-0.8B":        "name the model Vela-2.0-0.8B, without a repository or path",
		"Decision-2.0-Eos-0.8B":       "is a Decision 2.0 model. The built-in signals ask the questions a Vela model was trained on",
		"vllm-sr/Decision-2.0-Lux-9B": "name that deployment in a routing.signals.decision question",
		"Kai":                         "is a Decision 2.0 model",
		"nox-4b":                      "is a Decision 2.0 model",
	} {
		_, err := ParseYAMLBytes([]byte(decisionModelYAML("      decision_model: "+name+"\n", "")))
		if err == nil || !strings.Contains(err.Error(), "global.model_catalog.system.decision_model") || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: want an error containing %q, got %v", name, want, err)
		}
	}
}

func TestASystemLineKeepsItsModuleOffTheDecisionModel(t *testing.T) {
	raw := decisionModelYAML("      decision_model: Vela-2.0-0.8B\n      prompt_guard: models/Vela-1.0-Encoder-307M-Guard\n", "")
	cfg := mustParseDecisionModel(t, raw)
	if cfg.PromptGuard.ModelID != "models/Vela-1.0-Encoder-307M-Guard" || cfg.PromptGuard.Threshold != vela1ModuleThresholds.PromptGuard {
		t.Fatalf("prompt guard must keep its Vela 1.0 model and threshold, got %q at %v", cfg.PromptGuard.ModelID, cfg.PromptGuard.Threshold)
	}
	if cfg.CategoryModel.ModelID != Vela2Model08B || cfg.PIIModel.ModelID != Vela2Model08B {
		t.Fatalf("the other modules follow the decision model, got domain %q pii %q", cfg.CategoryModel.ModelID, cfg.PIIModel.ModelID)
	}
}

func TestAGPUOnlyDecisionModelRunsOnTheGPUWhateverUseCPUSays(t *testing.T) {
	modules := "      prompt_guard: {use_cpu: true}\n      classifier:\n        domain: {use_cpu: true}\n"
	got := deploymentsByArtifact(t, decisionModelYAML("      decision_model: Vela-2.0-9B\n", modules))
	if len(got) != 1 || len(got["vllm-sr/Vela-2.0-9B auto exact"]) != 1 {
		t.Fatalf("the 9B must run on one deployment on the best device, got %v", got)
	}
	if !RequiresGPU(Vela2Model4B) || !RequiresGPU(Vela2Model9B) || RequiresGPU(Vela2Model08B) || RequiresGPU(Vela2SignalModel) {
		t.Fatal("the 4B and 9B require a GPU; the 0.3B and 0.8B run on a CPU")
	}
}

func TestModuleThresholdsFollowTheDecisionModelUnlessSet(t *testing.T) {
	for _, name := range []string{DecisionModelVela2_03B, DecisionModelVela2_08B, DecisionModelVela2_4B, DecisionModelVela2_9B} {
		cfg := mustParseDecisionModel(t, decisionModelYAML("      decision_model: "+name+"\n", ""))
		spec, _ := LookupDecisionModel(name)
		want := vela2ModuleThresholds[spec.Model]
		got := ModuleThresholds{
			cfg.PromptGuard.Threshold, cfg.CategoryModel.Threshold, cfg.PIIModel.Threshold,
			cfg.HallucinationMitigation.FactCheckModel.Threshold, cfg.FeedbackDetector.Threshold,
		}
		if got != want {
			t.Fatalf("%s: module thresholds %+v, want %+v", name, got, want)
		}
	}
	cfg := mustParseDecisionModel(t, decisionModelYAML("      decision_model: Vela-2.0-9B\n", "      prompt_guard: {threshold: 0.42}\n"))
	if cfg.PromptGuard.Threshold != 0.42 {
		t.Fatalf("a threshold the configuration sets stays, got %v", cfg.PromptGuard.Threshold)
	}
}

// withDecisionQuestion adds a decision question that names no deployment.
func withDecisionQuestion(raw string) string {
	return strings.Replace(raw, "  signals:\n", "  signals:\n    decision:\n      - name: needs_tools\n"+
		"        question: {type: noul, instructions: \"Does the request need a tool?\"}\n", 1)
}

func TestADecisionQuestionWithoutDeploymentAsksTheDecisionModel(t *testing.T) {
	for system, want := range map[string]string{
		"":                                      "@Vela-2.0-0.3B",
		"      decision_model: Vela-2.0-0.8B\n": "@Vela-2.0-0.8B",
		"      decision_model: Vela-2.0-4B\n":   "@Vela-2.0-4B/auto",
	} {
		cfg := mustParseDecisionModel(t, withDecisionQuestion(decisionModelYAML(system, "")))
		if len(cfg.DecisionRules) != 1 {
			t.Fatalf("want one decision question, got %d", len(cfg.DecisionRules))
		}
		if got := cfg.DecisionQuestionDeployment(cfg.DecisionRules[0]); got != want {
			t.Fatalf("%q: the question asks %q, want %q", system, got, want)
		}
		if _, used := ModelRuntimeDeploymentsInUse(cfg)[want]; !used {
			t.Fatalf("%q: the decision model's deployment %s must be in use", system, want)
		}
	}
}

func TestADecisionQuestionOnAGPUAsksTheGPUDeployment(t *testing.T) {
	modules := "      prompt_guard: {use_cpu: false}\n"
	cfg := mustParseDecisionModel(t, withDecisionQuestion(decisionModelYAML("      decision_model: Vela-2.0-0.8B\n", modules)))
	if got := cfg.DecisionQuestionDeployment(cfg.DecisionRules[0]); got != "@Vela-2.0-0.8B/auto" {
		t.Fatalf("a module on the GPU puts the decision questions there, got %q", got)
	}
}

func TestADecisionQuestionWithoutDeploymentNeedsAVela2DecisionModel(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(withDecisionQuestion(decisionModelYAML("      decision_model: Vela-1.0\n", ""))))
	if err == nil || !strings.Contains(err.Error(), "routing.signals.decision[needs_tools]: deployment is required: the decision model is Vela-1.0") {
		t.Fatalf("want the Vela 1.0 deployment error, got %v", err)
	}
}

func TestExportWritesTheDecisionModelAndOnlyItsOverrides(t *testing.T) {
	raw := decisionModelYAML("      decision_model: vela-2.0-0.8b\n      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback\n", "")
	system := CanonicalConfigFromRouterConfig(mustParseDecisionModel(t, raw)).Global.ModelCatalog.System
	want := CanonicalSystemModels{DecisionModel: DecisionModelVela2_08B, FeedbackDetector: "models/Vela-1.0-Encoder-307M-Feedback"}
	if system != want {
		t.Fatalf("exported system %+v, want %+v", system, want)
	}
	if exported := CanonicalConfigFromRouterConfig(mustParseDecisionModel(t, builtInSignalsYAML(""))).Global.ModelCatalog.System; exported != (CanonicalSystemModels{}) {
		t.Fatalf("the default decision model exports no system lines, got %+v", exported)
	}
}

func TestTheReferenceConfigFollowsTheDecisionModel(t *testing.T) {
	data := readReferenceConfigYAML(t)
	raw := strings.Replace(string(data), "decision_model: Vela-2.0-0.3B", "decision_model: Vela-2.0-9B", 1)
	if raw == string(data) {
		t.Fatal("config/config.yaml must name the decision model")
	}
	cfg := mustParseDecisionModel(t, raw)
	for module, model := range moduleModels(cfg) {
		if model != Vela2Model9B {
			t.Fatalf("%s runs %q; the reference config must let every module follow the decision model", module, model)
		}
	}
	want := vela2ModuleThresholds[Vela2Model9B]
	if cfg.PromptGuard.Threshold != want.PromptGuard || cfg.CategoryModel.Threshold != want.Domain || cfg.PIIModel.Threshold != want.PII ||
		cfg.HallucinationMitigation.FactCheckModel.Threshold != want.FactCheck || cfg.FeedbackDetector.Threshold != want.Feedback {
		t.Fatal("the reference config must leave the module thresholds to the decision model")
	}
}
