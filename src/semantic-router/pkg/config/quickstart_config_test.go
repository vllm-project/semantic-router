package config

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

// config/quickstart.yaml is the maintained on-ramp config (#2690): the
// smallest config that passes validation, routes with model-free signals
// only, and requires no classifier-model artifacts. These tests are the
// regression gate that keeps it that way.

func readQuickstartConfigYAML(t testingT) []byte {
	t.Helper()
	root := referenceConfigRepoRoot(t)
	path := filepath.Join(root, "config", "quickstart.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", path, err)
	}
	return data
}

func TestQuickstartConfigUsesStrictCanonicalSchema(t *testing.T) {
	data := readQuickstartConfigYAML(t)

	decoder := yamlv3.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)

	var canonical CanonicalConfig
	if err := decoder.Decode(&canonical); err != nil {
		t.Fatalf("config/quickstart.yaml no longer matches the strict canonical schema: %v", err)
	}

	if canonical.Version != "v0.3" {
		t.Fatalf("expected quickstart config version v0.3, got %q", canonical.Version)
	}

	if _, err := ParseYAMLBytes(data); err != nil {
		t.Fatalf("config/quickstart.yaml failed runtime parse validation: %v", err)
	}
}

func TestQuickstartConfigRequiresNoModelArtifacts(t *testing.T) {
	cfg, err := ParseYAMLBytes(readQuickstartConfigYAML(t))
	if err != nil {
		t.Fatalf("config/quickstart.yaml failed runtime parse validation: %v", err)
	}

	for _, path := range []struct {
		name  string
		value string
	}{
		{"mmbert_model_path", cfg.MmBertModelPath},
		{"qwen3_model_path", cfg.Qwen3ModelPath},
		{"gemma_model_path", cfg.GemmaModelPath},
		{"bert_model_path", cfg.BertModelPath},
		{"multimodal_model_path", cfg.MultiModalModelPath},
	} {
		if path.value != "" {
			t.Errorf("quickstart config must not set embedding %s, got %q", path.name, path.value)
		}
	}

	// These are the same predicates pkg/modeldownload uses to decide which
	// model artifacts a config requires; all must stay off for quickstart.
	for name, required := range map[string]bool{
		"category classifier":   cfg.NeedsCategoryMappingForRouting(),
		"pii classifier":        cfg.NeedsPIIMappingForRouting(),
		"jailbreak classifier":  cfg.NeedsJailbreakMappingForRouting(),
		"fact-check classifier": cfg.IsFactCheckClassifierEnabled(),
		"hallucination model":   cfg.IsHallucinationModelEnabled(),
		"feedback detector":     cfg.IsFeedbackDetectorEnabled(),
	} {
		if required {
			t.Errorf("quickstart config must not require the %s model", name)
		}
	}
}

func TestQuickstartConfigDemonstratesDistinctRouting(t *testing.T) {
	var root map[string]interface{}
	if err := yamlv3.Unmarshal(readQuickstartConfigYAML(t), &root); err != nil {
		t.Fatalf("failed to unmarshal config/quickstart.yaml into raw map: %v", err)
	}

	declaredSignals := map[string]bool{}
	for _, rawSignal := range mustSliceAt(t, root, "routing", "signals", "keywords") {
		declaredSignals[mustStringAt(t, mustMapValue(t, rawSignal, "routing.signals.keywords[]"), "name")] = true
	}

	decisions := mustSliceAt(t, root, "routing", "decisions")
	if len(decisions) < 2 {
		t.Fatalf("quickstart config must keep at least two decisions to demonstrate routing, got %d", len(decisions))
	}

	selectedModels := map[string]bool{}
	sawConditionedDecision := false
	for _, rawDecision := range decisions {
		decision := mustMapValue(t, rawDecision, "routing.decisions[]")
		name := mustStringAt(t, decision, "name")

		// Every decision must complete without an LLM backend.
		if !decisionHasPluginType(t, decision, "fast_response") {
			t.Errorf("decision %q must carry a fast_response plugin so quickstart completes without a backend", name)
		}

		for _, rawRef := range mustSliceValue(t, decision["modelRefs"], "routing.decisions[].modelRefs") {
			selectedModels[mustStringAt(t, mustMapValue(t, rawRef, "modelRefs[]"), "model")] = true
		}

		rules := mustMapValue(t, decision["rules"], "routing.decisions[].rules")
		for _, rawCondition := range mustSliceValue(t, rules["conditions"], "rules.conditions") {
			condition := mustMapValue(t, rawCondition, "rules.conditions[]")
			if mustStringAt(t, condition, "type") != "keyword" {
				t.Errorf("decision %q uses a non-keyword condition; quickstart must stay model-free", name)
				continue
			}
			sawConditionedDecision = true
			if signal := mustStringAt(t, condition, "name"); !declaredSignals[signal] {
				t.Errorf("decision %q references undeclared keyword signal %q", name, signal)
			}
		}
	}

	if !sawConditionedDecision {
		t.Fatalf("quickstart config must keep at least one keyword-conditioned decision")
	}
	if len(selectedModels) < 2 {
		t.Fatalf("quickstart config must select at least two distinct models across decisions, got %v", selectedModels)
	}
}

func decisionHasPluginType(t testingT, decision map[string]interface{}, pluginType string) bool {
	t.Helper()
	rawPlugins, ok := decision["plugins"]
	if !ok || rawPlugins == nil {
		return false
	}
	for _, rawPlugin := range mustSliceValue(t, rawPlugins, "routing.decisions[].plugins") {
		plugin := mustMapValue(t, rawPlugin, "plugins[]")
		if typ, ok := plugin["type"].(string); ok && typ == pluginType {
			return true
		}
	}
	return false
}
