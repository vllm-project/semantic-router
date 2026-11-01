package modelruntimevela3

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
					Question   struct {
						Type string `yaml:"type"`
					} `yaml:"question"`
				} `yaml:"decision"`
			} `yaml:"signals"`
			Decisions []struct {
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
					Device   string `yaml:"device"`
				} `yaml:"deployments"`
				System struct {
					DecisionModel struct {
						Deployment string `yaml:"deployment"`
					} `yaml:"decision_model"`
				} `yaml:"system"`
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

// The names testcases/model_runtime_vela3.go relies on.
func TestProfileServesPinnedVela3FastAsTheDecisionModel(t *testing.T) {
	profile := load(t)
	catalog := profile.Config.Global.ModelCatalog
	primary, ok := catalog.Deployments["primary"]
	if !ok || primary.Provider != "model_runtime" || primary.Endpoint != "" || primary.Device != "cpu" {
		t.Fatalf("primary must be a managed cpu deployment: %+v", primary)
	}
	if primary.Artifact != "vllm-sr/vela3-fast" || !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(primary.Revision) {
		t.Fatalf("primary must pin the published vela3-fast revision: %+v", primary)
	}
	if len(catalog.Deployments) != 1 || catalog.System.DecisionModel.Deployment != "primary" {
		t.Fatal("one vela3-fast deployment must answer every signal")
	}
	kinds := map[string]string{}
	for _, rule := range profile.Config.Routing.Signals.Decision {
		if rule.Deployment != "primary" {
			t.Fatalf("decision signal %s must ask primary", rule.Name)
		}
		kinds[rule.Name] = rule.Question.Type
	}
	if kinds["account_access"] != "noul" || kinds["order_numbers"] != "span" {
		t.Fatalf("the cases expect a Noul and a Span decision signal: %v", kinds)
	}
	decisions := map[string]bool{}
	for _, decision := range profile.Config.Routing.Decisions {
		decisions[decision.Name] = true
	}
	for _, name := range []string{"jailbreak_route", "pii_route", "order_route", "account_route", "default-route"} {
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
		t.Fatalf("the cases reach vela3-fast through /tmp/vsr-runtime: %v", env)
	}
	for _, name := range []string{"VLLM_SRUN_COMMAND", "HF_HUB_OFFLINE"} {
		if _, set := env[name]; set {
			t.Fatalf("%s would replace the real runtime or block its downloads", name)
		}
	}
}
