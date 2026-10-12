package handlers

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

type adaptationsRouting struct {
	Decisions []struct {
		Name        string                                 `yaml:"name"`
		Adaptations routerconfig.DecisionAdaptationsConfig `yaml:"adaptations"`
	} `yaml:"decisions"`
}

// decisionAdaptations maps each decision, as name or recipe/name, to its adaptations.
func decisionAdaptations(t *testing.T, raw []byte) map[string]routerconfig.DecisionAdaptationsConfig {
	t.Helper()
	var doc struct {
		Routing adaptationsRouting `yaml:"routing"`
		Recipes []struct {
			Name    string             `yaml:"name"`
			Routing adaptationsRouting `yaml:"routing"`
		} `yaml:"recipes"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("decode decisions: %v", err)
	}
	found := map[string]routerconfig.DecisionAdaptationsConfig{}
	for _, decision := range doc.Routing.Decisions {
		found[decision.Name] = decision.Adaptations
	}
	for _, recipe := range doc.Recipes {
		for _, decision := range recipe.Routing.Decisions {
			found[recipe.Name+"/"+decision.Name] = decision.Adaptations
		}
	}
	return found
}

func assertDecisionAdaptations(t *testing.T, merged []byte, want map[string]routerconfig.DecisionAdaptationsConfig) {
	t.Helper()
	got := decisionAdaptations(t, merged)
	var changed []string
	for name := range got {
		if _, known := want[name]; !known {
			changed = append(changed, name)
		}
	}
	for name, value := range want {
		if actual, ok := got[name]; !ok || !reflect.DeepEqual(actual, value) {
			changed = append(changed, name)
		}
	}
	if len(changed) > 0 {
		sort.Strings(changed)
		t.Fatalf("adaptations differ after deploy for decisions %v", changed)
	}
}

func TestMergeDeployPayloadKeepsBaseDecisionAdaptations(t *testing.T) {
	base := `routing:
  decisions:
    - name: hard_policy
      priority: 10
      adaptations:
        mode: bypass
    - name: plain
      priority: 5
recipes:
  - name: alpha
    routing:
      decisions:
        - name: shared
          priority: 1
          adaptations: {mode: observe}
  - name: beta
    routing:
      decisions:
        - name: shared
          priority: 1
          adaptations:
            adaptation: {candidate_set: decision}
`
	// A compiled DSL fragment never carries adaptations.
	fragment := `routing:
  decisions:
    - name: hard_policy
      priority: 20
    - name: plain
      priority: 5
    - name: added
      priority: 1
recipes:
  - name: alpha
    routing:
      decisions:
        - name: shared
          priority: 2
  - name: beta
    routing:
      decisions:
        - name: shared
          priority: 1
          adaptations: {mode: apply}
  - name: gamma
    routing:
      decisions:
        - name: shared
          priority: 1
`
	want := map[string]routerconfig.DecisionAdaptationsConfig{
		"hard_policy":  {Mode: routerconfig.DecisionAdaptationModeBypass},
		"plain":        {},
		"added":        {},
		"alpha/shared": {Mode: routerconfig.DecisionAdaptationModeObserve},
		"beta/shared":  {Mode: routerconfig.DecisionAdaptationModeApply},
		"gamma/shared": {},
	}

	for _, mode := range []DeployMode{DeployModeReplace, DeployModeMerge} {
		t.Run(string(mode), func(t *testing.T) {
			merged, err := mergeDeployPayload([]byte(base), DeployRequest{YAML: fragment, Mode: mode})
			if err != nil {
				t.Fatalf("mergeDeployPayload error: %v", err)
			}
			assertDecisionAdaptations(t, merged, want)
		})
	}
}

// The Builder imports a config, decompiles it to DSL, compiles it back, and
// deploys the compiled fragment in replace mode.
func TestMergeDeployPayloadReplaceKeepsMaintainedDecisionAdaptations(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	// Immutable release snapshots such as built-in/v0.4 may not parse with the current schema.
	for _, relative := range []string{
		"config/config.yaml",
		"config/recipes/agent/config.yaml",
		"config/recipes/built-in/latest/mom-v1/config.yaml",
	} {
		t.Run(relative, func(t *testing.T) {
			original, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(relative)))
			if err != nil {
				t.Fatalf("read config: %v", err)
			}
			want := decisionAdaptations(t, original)
			if !hasDecisionAdaptations(want) {
				t.Fatalf("%s has no decision adaptations to check", relative)
			}

			cfg, err := routerconfig.ParseYAMLBytes(original)
			if err != nil {
				t.Fatalf("ParseYAMLBytes error: %v", err)
			}
			text, err := dsl.Decompile(cfg)
			if err != nil {
				t.Fatalf("Decompile error: %v", err)
			}
			compiled, errs := dsl.Compile(text)
			if len(errs) > 0 {
				t.Fatalf("Compile errors: %v", errs)
			}
			fragment, err := dsl.EmitRoutingYAMLFromConfig(compiled)
			if err != nil {
				t.Fatalf("EmitRoutingYAMLFromConfig error: %v", err)
			}

			merged, err := mergeDeployPayload(original, DeployRequest{
				YAML:     string(fragment),
				BaseYAML: string(original),
				Mode:     DeployModeReplace,
			})
			if err != nil {
				t.Fatalf("mergeDeployPayload error: %v", err)
			}
			assertDecisionAdaptations(t, merged, want)
		})
	}
}

func hasDecisionAdaptations(adaptations map[string]routerconfig.DecisionAdaptationsConfig) bool {
	for _, value := range adaptations {
		if !reflect.DeepEqual(value, routerconfig.DecisionAdaptationsConfig{}) {
			return true
		}
	}
	return false
}
