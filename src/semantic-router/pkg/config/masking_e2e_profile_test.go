package config

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

const (
	maskingProfileValues     = "e2e/profiles/masking/values.yaml"
	maskingProfileEntrypoint = "vllm-sr/masking"
	maskingProfileDecision   = "mask_pii"
)

// The masking e2e profile first shipped pointing at a concrete model name,
// which goes through specified-model routing and bypasses every recipe-local
// decision and plugin: the plugin never ran, so the cluster assertions would
// have passed while masking nothing. This pins the chain that profile depends
// on -- the entrypoint resolves to a recipe, that recipe carries the masking
// decision, and masking is enabled on it -- here, where it costs a second,
// rather than discovering it from a cluster run.
func TestMaskingE2EProfileEntrypointReachesMaskingDecision(t *testing.T) {
	root := repoRootFromTestFile(t)
	valuesPath := filepath.Join(root, maskingProfileValues)

	raw, err := os.ReadFile(valuesPath)
	if err != nil {
		t.Fatalf("read %s: %v", valuesPath, err)
	}
	var values struct {
		Config map[string]interface{} `yaml:"config"`
	}
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatalf("parse %s: %v", valuesPath, err)
	}
	if len(values.Config) == 0 {
		t.Fatalf("%s declares no config block", valuesPath)
	}
	encoded, err := yaml.Marshal(values.Config)
	if err != nil {
		t.Fatalf("re-encode config block: %v", err)
	}

	routerConfig, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("%s config block is not a valid router config: %v", valuesPath, err)
	}

	recipe, ok := routerConfig.RecipeForRoutingModel(maskingProfileEntrypoint)
	if !ok || recipe == nil {
		t.Fatalf(
			"entrypoint %q resolves to no recipe, so the request would bypass its decision and mask nothing",
			maskingProfileEntrypoint,
		)
	}

	var decision *Decision
	for i := range recipe.Profile.Decisions {
		if recipe.Profile.Decisions[i].Name == maskingProfileDecision {
			decision = &recipe.Profile.Decisions[i]
			break
		}
	}
	if decision == nil {
		t.Fatalf("recipe %q has no decision %q", recipe.Name, maskingProfileDecision)
	}
	masking := decision.GetMaskingConfig()
	if masking == nil {
		t.Fatalf("decision %q does not configure the masking plugin", maskingProfileDecision)
	}
	if !masking.Enabled {
		t.Fatalf("decision %q configures masking but leaves it disabled", maskingProfileDecision)
	}
}
