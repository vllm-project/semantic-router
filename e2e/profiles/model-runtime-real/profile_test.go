package modelruntimereal

import (
	"os"
	"regexp"
	"testing"

	"gopkg.in/yaml.v3"
)

type values struct {
	Config struct {
		Routing struct {
			Signals struct {
				Decision []struct {
					Name       string `yaml:"name"`
					Deployment string `yaml:"deployment"`
				} `yaml:"decision"`
			} `yaml:"signals"`
			ModelBindings map[string]any `yaml:"model_bindings"`
			Decisions     []struct {
				Name string `yaml:"name"`
			} `yaml:"decisions"`
		} `yaml:"routing"`
		Global struct {
			ModelCatalog struct {
				Deployments map[string]struct {
					Provider string `yaml:"provider"`
					Artifact string `yaml:"artifact"`
					Revision string `yaml:"revision"`
					Endpoint string `yaml:"endpoint"`
					Process  string `yaml:"process"`
					Device   string `yaml:"device"`
				} `yaml:"deployments"`
				Bindings map[string]any `yaml:"bindings"`
			} `yaml:"model_catalog"`
		} `yaml:"global"`
	} `yaml:"config"`
	ExtraEnv []struct {
		Name  string `yaml:"name"`
		Value string `yaml:"value"`
	} `yaml:"extraEnv"`
}

func load(t *testing.T) values {
	t.Helper()
	data, err := os.ReadFile("values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var profile values
	if err := yaml.Unmarshal(data, &profile); err != nil {
		t.Fatal(err)
	}
	return profile
}

// The names testcases/model_runtime_real.go relies on.
func TestProfileServesRealKaiAndTheVelaDefaults(t *testing.T) {
	profile := load(t)
	kai, ok := profile.Config.Global.ModelCatalog.Deployments["kai"]
	if !ok || kai.Provider != "model_runtime" || kai.Endpoint != "" || kai.Device != "cpu" || kai.Process != "decisions" {
		t.Fatalf("kai must be a managed cpu deployment in the decisions process: %+v", kai)
	}
	if kai.Artifact != "vllm-sr/Decision-2.0-Kai-0.6B" || !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(kai.Revision) {
		t.Fatalf("kai must pin the published Kai-0.6B revision: %+v", kai)
	}
	if len(profile.Config.Global.ModelCatalog.Deployments) != 1 {
		t.Fatal("the Vela models must come from the module defaults, not declared deployments")
	}
	if len(profile.Config.Global.ModelCatalog.Bindings) != 0 || len(profile.Config.Routing.ModelBindings) != 0 {
		t.Fatal("bindings would replace the implicit @<module> deployments the cases wait for")
	}
	signals := map[string]string{}
	for _, rule := range profile.Config.Routing.Signals.Decision {
		signals[rule.Name] = rule.Deployment
	}
	if signals["request_kind"] != "kai" {
		t.Fatalf("request_kind must ask kai: %v", signals)
	}
	decisions := map[string]bool{}
	for _, decision := range profile.Config.Routing.Decisions {
		decisions[decision.Name] = true
	}
	for _, name := range []string{"jailbreak_route", "pii_route", "code_route", "math_route", "default-route"} {
		if !decisions[name] {
			t.Fatalf("decision %s is missing", name)
		}
	}
}

func TestProfileRunsTheRealRuntimeWithKnownSockets(t *testing.T) {
	env := map[string]string{}
	for _, item := range load(t).ExtraEnv {
		env[item.Name] = item.Value
	}
	if env["VLLM_SRUN_DIR"] != "/tmp/vsr-runtime" {
		t.Fatalf("the cases reach Kai through /tmp/vsr-runtime: %v", env)
	}
	for _, name := range []string{"VLLM_SRUN_COMMAND", "HF_HUB_OFFLINE"} {
		if _, set := env[name]; set {
			t.Fatalf("%s would replace the real runtime or block its downloads", name)
		}
	}
}
