package modelruntime

import (
	"os"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// values mirrors the parts of values.yaml this test checks.
type values struct {
	Config struct {
		Recipes []struct {
			Signals struct {
				Keywords []struct {
					Name     string   `yaml:"name"`
					Method   string   `yaml:"method"`
					Keywords []string `yaml:"keywords"`
				} `yaml:"keywords"`
				SystemOne []struct {
					Name       string                 `yaml:"name"`
					Deployment string                 `yaml:"deployment"`
					Question   map[string]interface{} `yaml:"question"`
					Predicate  map[string]float64     `yaml:"predicate"`
				} `yaml:"systemone"`
			} `yaml:"signals"`
			ModelBindings map[string]string `yaml:"model_bindings"`
			Decisions     []struct {
				Name     string `yaml:"name"`
				Priority int    `yaml:"priority"`
				Rules    struct {
					Operator   string `yaml:"operator"`
					OnUnknown  string `yaml:"on_unknown"`
					Conditions []struct {
						Type string `yaml:"type"`
						Name string `yaml:"name"`
					} `yaml:"conditions"`
				} `yaml:"rules"`
				ModelRefs []struct {
					Model string `yaml:"model"`
				} `yaml:"modelRefs"`
				Algorithm *struct {
					SystemOne *struct {
						Deployment string `yaml:"deployment"`
					} `yaml:"systemone"`
				} `yaml:"algorithm"`
			} `yaml:"decisions"`
		} `yaml:"recipes"`
		Global struct {
			ModelCatalog struct {
				Deployments map[string]struct {
					Provider string `yaml:"provider"`
					Artifact string `yaml:"artifact"`
					Endpoint string `yaml:"endpoint"`
					Model    string `yaml:"model"`
					Device   string `yaml:"device"`
				} `yaml:"deployments"`
				Bindings   map[string]string `yaml:"bindings"`
				Embeddings struct {
					Default struct {
						Deployment string `yaml:"deployment"`
					} `yaml:"default"`
				} `yaml:"embeddings"`
			} `yaml:"model_catalog"`
		} `yaml:"global"`
	} `yaml:"config"`
	ExtraEnv []struct {
		Name  string `yaml:"name"`
		Value string `yaml:"value"`
	} `yaml:"extraEnv"`
	ExtraVolumes []struct {
		ConfigMap struct {
			Name string `yaml:"name"`
		} `yaml:"configMap"`
	} `yaml:"extraVolumes"`
}

func load(t *testing.T, path string, into interface{}) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(data, into); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// fixturePackages lists the package directories runtime_with_fixtures.py writes.
func fixturePackages(t *testing.T) map[string]bool {
	t.Helper()
	source, err := os.ReadFile("runtime_with_fixtures.py")
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)PACKAGES = \{(.*?)\n\}`).FindSubmatch(source)
	if block == nil {
		t.Fatal("runtime_with_fixtures.py has no PACKAGES table")
	}
	packages := map[string]bool{}
	for _, match := range regexp.MustCompile(`"([\w-]+)": \(`).FindAllSubmatch(block[1], -1) {
		packages[string(match[1])] = true
	}
	return packages
}

func TestProfileRunsManagedAndAttachedRuntimesOnItsOwnFixtures(t *testing.T) {
	var profile values
	load(t, "values.yaml", &profile)
	packages := fixturePackages(t)
	deployments := profile.Config.Global.ModelCatalog.Deployments

	// The names testcases/model_runtime_support.go relies on, with their
	// device: the lifecycle case checks independent workers, including auto
	// resolving to CPU on a node without an accelerator.
	managed := map[string]struct{ device string }{
		"decision-fixture": {"cpu"}, "vela-embedding": {"auto"}, "vela31-fixture": {"cpu"},
	}
	for name, want := range managed {
		deployment, ok := deployments[name]
		if !ok || deployment.Provider != "model_runtime" || deployment.Endpoint != "" || deployment.Device != want.device {
			t.Fatalf("%s must be a managed %s deployment: %+v", name, want.device, deployment)
		}
		directory := strings.TrimPrefix(deployment.Artifact, "/tmp/vsr-fixtures/")
		if !packages[directory] {
			t.Fatalf("%s names %s, which runtime_with_fixtures.py does not write", name, deployment.Artifact)
		}
	}
	attached := map[string]string{"attached-decisions": "decision-a", "attached-vela2": "vela2-a"}
	for name, served := range attached {
		deployment := deployments[name]
		if !strings.Contains(deployment.Endpoint, "model-runtime-attached.") || deployment.Model != served {
			t.Fatalf("%s must attach to model-runtime-attached as %s: %+v", name, served, deployment)
		}
	}
	if offline := deployments["decision-offline"]; offline.Endpoint == "" || strings.Contains(offline.Endpoint, "model-runtime-attached") {
		t.Fatalf("decision-offline must name an endpoint nothing serves: %+v", offline)
	}

	var models struct {
		Models []struct {
			Model string `yaml:"model"`
			Name  string `yaml:"name"`
		} `yaml:"models"`
	}
	load(t, "attached-models.yaml", &models)
	served := map[string]bool{}
	for _, model := range models.Models {
		served[model.Name] = true
		if !packages[strings.TrimPrefix(model.Model, "/tmp/vsr-fixtures/")] {
			t.Fatalf("attached model %s names %s, which runtime_with_fixtures.py does not write", model.Name, model.Model)
		}
	}
	for _, name := range attached {
		if !served[name] {
			t.Fatalf("the attached runtime does not serve %s", name)
		}
	}

	used := map[string]bool{profile.Config.Global.ModelCatalog.Embeddings.Default.Deployment: true}
	for _, deployment := range profile.Config.Global.ModelCatalog.Bindings {
		used[deployment] = true
	}
	for _, deployment := range profile.Config.Recipes[0].ModelBindings {
		used[deployment] = true
	}
	for _, rule := range profile.Config.Recipes[0].Signals.SystemOne {
		used[rule.Deployment] = true
	}
	for _, decision := range profile.Config.Recipes[0].Decisions {
		if decision.Algorithm != nil && decision.Algorithm.SystemOne != nil {
			used[decision.Algorithm.SystemOne.Deployment] = true
		}
	}
	delete(used, "")
	for name := range deployments {
		if !used[name] {
			t.Fatalf("deployment %s is declared but nothing uses it, so the Router would not start it", name)
		}
	}
	for name := range used {
		if _, ok := deployments[name]; !ok {
			t.Fatalf("a consumer names undeclared deployment %s", name)
		}
	}
}

func TestProfileKeepsVela31DescribedNoulOnAMarkerSpecificRoute(t *testing.T) {
	cases := NewProfile().GetTestCases()
	if len(cases) != 11 || !slices.Contains(cases, "model-runtime-vela31-described-noul") {
		t.Fatalf("profile must retain its ten contracts and the Vela 3.1 regression: %v", cases)
	}
	var profile values
	load(t, "values.yaml", &profile)
	recipe := profile.Config.Recipes[0]
	foundSignal, foundRoute, foundMarker := false, false, false
	for _, signal := range recipe.Signals.SystemOne {
		if signal.Name != "vela31_described" {
			continue
		}
		foundSignal = true
		want := map[string]interface{}{
			"type": "noul", "instructions": "Does this request need urgent action?",
			"choices": []interface{}{
				map[string]interface{}{"key": "false", "description": "can wait"},
				map[string]interface{}{"key": "true", "description": "needs immediate action"},
			},
		}
		if signal.Deployment != "vela31-fixture" || !reflect.DeepEqual(signal.Question, want) ||
			!reflect.DeepEqual(signal.Predicate, map[string]float64{"gte": 0.0001}) {
			t.Fatalf("Vela 3.1 must ask the described-Noul contract with a nonzero predicate: %+v", signal)
		}
	}
	for _, marker := range recipe.Signals.Keywords {
		if marker.Name != "vela31_marker" {
			continue
		}
		foundMarker = true
		if marker.Method != "regex" || len(marker.Keywords) != 1 {
			t.Fatalf("marker must be one explicit regex: %+v", marker)
		}
		pattern := regexp.MustCompile(marker.Keywords[0])
		if !pattern.MatchString("[vela31-described-noul] Please refund my duplicate charge.") ||
			pattern.MatchString("Write a Go function that merges two sorted slices.") {
			t.Fatalf("marker must distinguish the regression prompt from existing selector prompts")
		}
	}
	for _, decision := range recipe.Decisions {
		if decision.Name != "vela31_described_route" {
			continue
		}
		foundRoute = true
		conditions := map[string]string{}
		for _, condition := range decision.Rules.Conditions {
			conditions[condition.Type] = condition.Name
		}
		if decision.Priority <= 500 || decision.Rules.Operator != "AND" || decision.Rules.OnUnknown != "no_match" ||
			len(decision.Rules.Conditions) != 2 || !reflect.DeepEqual(conditions, map[string]string{"keyword": "vela31_marker", "systemone": "vela31_described"}) ||
			len(decision.ModelRefs) != 1 || decision.ModelRefs[0].Model != "math-expert" || decision.Algorithm != nil {
			t.Fatalf("described Noul and marker must select math-expert without the old selector: %+v", decision)
		}
	}
	if !foundSignal || !foundRoute || !foundMarker {
		t.Fatalf("missing Vela 3.1 profile binding: signal=%v route=%v marker=%v", foundSignal, foundRoute, foundMarker)
	}
}

func TestRouterStartsManagedRuntimesThroughTheFixtureScript(t *testing.T) {
	var profile values
	load(t, "values.yaml", &profile)
	env := map[string]string{}
	for _, item := range profile.ExtraEnv {
		env[item.Name] = item.Value
	}
	if env["VLLM_SRUN_COMMAND"] != "python3 /opt/vsr-e2e/runtime_with_fixtures.py" || env["VLLM_SRUN_DIR"] != "/tmp/vsr-runtime" {
		t.Fatalf("managed runtimes must start through the fixture script with a known socket directory: %v", env)
	}
	if len(profile.ExtraVolumes) != 1 || profile.ExtraVolumes[0].ConfigMap.Name != FilesConfigMap {
		t.Fatalf("the Router pod must mount the %s ConfigMap: %+v", FilesConfigMap, profile.ExtraVolumes)
	}
	for key, path := range configMapFiles {
		if _, err := os.Stat(strings.TrimPrefix(path, profileDir)); err != nil {
			t.Fatalf("ConfigMap key %s: %v", key, err)
		}
	}
}
