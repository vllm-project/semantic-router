package modelruntime

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// values mirrors the parts of values.yaml this test checks.
type values struct {
	Config struct {
		Routing struct {
			Signals struct {
				Domains  []struct{ Name string } `yaml:"domains"`
				Decision []struct {
					Name       string `yaml:"name"`
					Deployment string `yaml:"deployment"`
				} `yaml:"decision"`
			} `yaml:"signals"`
			ModelBindings map[string]struct {
				Deployment string `yaml:"deployment"`
			} `yaml:"model_bindings"`
			Decisions []struct {
				Name      string `yaml:"name"`
				Algorithm *struct {
					Decision *struct {
						Deployment string `yaml:"deployment"`
					} `yaml:"decision"`
				} `yaml:"algorithm"`
			} `yaml:"decisions"`
		} `yaml:"routing"`
		Global struct {
			ModelCatalog struct {
				Deployments map[string]struct {
					Provider   string `yaml:"provider"`
					Artifact   string `yaml:"artifact"`
					Endpoint   string `yaml:"endpoint"`
					ServedName string `yaml:"served_name"`
					Process    string `yaml:"process"`
					Device     string `yaml:"device"`
				} `yaml:"deployments"`
				Bindings map[string]struct {
					Deployment string `yaml:"deployment"`
				} `yaml:"bindings"`
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
	// process and device: the lifecycle case requires the one on auto to run
	// in the cpu device group.
	managed := map[string]struct{ process, device string }{
		"decision-fixture": {"decisions", "cpu"}, "vela-domain": {"", "cpu"}, "vela-pii": {"", "cpu"}, "vela-guard": {"", "cpu"},
		"vela-embedding": {"", "auto"}, "vela-reranker": {"", "cpu"}, "vela-modality": {"", "cpu"},
	}
	for name, want := range managed {
		deployment, ok := deployments[name]
		if !ok || deployment.Provider != "model_runtime" || deployment.Endpoint != "" || deployment.Device != want.device || deployment.Process != want.process {
			t.Fatalf("%s must be a managed %s deployment in process %q: %+v", name, want.device, want.process, deployment)
		}
		directory := strings.TrimPrefix(deployment.Artifact, "/tmp/vsr-fixtures/")
		if !packages[directory] {
			t.Fatalf("%s names %s, which runtime_with_fixtures.py does not write", name, deployment.Artifact)
		}
	}
	attached := map[string]string{"attached-decisions": "decision-a", "attached-feedback": "feedback-a", "attached-vela2": "vela2-a"}
	for name, served := range attached {
		deployment := deployments[name]
		if !strings.Contains(deployment.Endpoint, "model-runtime-attached.") || deployment.ServedName != served {
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

	used := map[string]bool{}
	for _, binding := range profile.Config.Global.ModelCatalog.Bindings {
		used[binding.Deployment] = true
	}
	for _, binding := range profile.Config.Routing.ModelBindings {
		used[binding.Deployment] = true
	}
	for _, rule := range profile.Config.Routing.Signals.Decision {
		used[rule.Deployment] = true
	}
	for _, decision := range profile.Config.Routing.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Decision != nil {
			used[decision.Algorithm.Decision.Deployment] = true
		}
	}
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
	if len(profile.Config.Routing.Signals.Domains) != 14 {
		t.Fatalf("one domain rule per domain fixture label is needed, got %d", len(profile.Config.Routing.Signals.Domains))
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

// The modules name no mapping file, so the lane covers labels taken from the
// served cards; a mapping file would also be a Hub fetch HF_HUB_OFFLINE forbids.
func TestModulesTakeTheirLabelsFromTheServedCards(t *testing.T) {
	var profile struct {
		Config struct {
			Global struct {
				ModelCatalog struct {
					Modules struct {
						Classifier struct {
							Domain struct {
								Mapping   string  `yaml:"category_mapping_path"`
								Threshold float64 `yaml:"threshold"`
							} `yaml:"domain"`
							PII struct {
								Mapping string `yaml:"pii_mapping_path"`
							} `yaml:"pii"`
						} `yaml:"classifier"`
						PromptGuard struct {
							Mapping string `yaml:"jailbreak_mapping_path"`
						} `yaml:"prompt_guard"`
					} `yaml:"modules"`
				} `yaml:"model_catalog"`
			} `yaml:"global"`
		} `yaml:"config"`
	}
	load(t, "values.yaml", &profile)
	modules := profile.Config.Global.ModelCatalog.Modules
	// testcases/model_runtime_signals.go (mrDomainThreshold) checks matches against it.
	if modules.Classifier.Domain.Threshold != 0.01 {
		t.Fatalf("domain threshold %v, the task-signal case expects 0.01", modules.Classifier.Domain.Threshold)
	}
	for _, path := range []string{modules.Classifier.Domain.Mapping, modules.Classifier.PII.Mapping, modules.PromptGuard.Mapping} {
		if path != "" {
			t.Fatalf("label map %q: the modules must take their labels from the served cards", path)
		}
	}
}
