package config

import (
	"reflect"
	"testing"

	"gopkg.in/yaml.v2"
)

const defaultHazardConsumerYAML = `version: v0.3
entrypoints:
  - model_names: [care]
    recipe: care
recipes:
  - name: care
    routing:
      model_bindings:
        classifier.content-risk:
          deployment: hazard
          adapter: modernbert
          contract: label_scores.v1
          operating_point:
            path: operating_point.json
            sha256: e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5
      signals:
        classifiers:
          - name: content-risk
            type: local
            labels: [violence, criminal_activity, sexual_content, child_exploitation, hate, harassment_abuse, regulated_substances, weapons, self_harm, privacy, specialized_advice, misinformation]
`

func TestDefaultHazardDeploymentIsPinnedAndOptIn(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte("version: v0.3\nrouting: {}\n"))
	if err != nil {
		t.Fatal(err)
	}
	deployment, ok := cfg.ModelDeployments["hazard"]
	if !ok {
		t.Fatal("missing logical Hazard deployment")
	}
	model := GetModelByPath("models/Vela-1.0-Encoder-307M-Hazard")
	if model == nil || len(model.Revision) != 40 {
		t.Fatal("Hazard default has no pinned registry artifact")
	}
	want := ModelDeployment{
		Artifact: model.LocalPath, Revision: model.Revision,
		Provider: "candle", Device: "cpu", Precision: "fp32",
		Input: ModelInputBudget{MaxTokens: 32768, Overflow: "reject"},
	}
	if deployment != want {
		t.Fatalf("deployment=%+v, want %+v", deployment, want)
	}
	if cfg.SafetyModels.Hazard.ModelID != "" || len(cfg.ClassifierRules) != 0 || len(cfg.ModelBindings) != 0 {
		t.Fatal("declaring a Hazard deployment activated a consumer")
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Global.ModelCatalog.Deployments["hazard"] != want {
		t.Fatal("canonical export lost default deployment")
	}
}

func TestDefaultHazardNamedBindingRoundTripAndOverride(t *testing.T) {
	for _, amd := range []bool{false, true} {
		name := "portable"
		raw := defaultHazardConsumerYAML
		if amd {
			name = "explicit_amd"
			raw += `global:
  model_catalog:
    deployments:
      hazard:
        artifact: models/Vela-1.0-Encoder-307M-Hazard
        revision: 5dd25f2cc3c98f338e6a79b667662d60f936a28d
        provider: ort
        device: migraphx:0
        precision: native
        compilation_cache_dir: /tmp/router-compilation-cache
        input:
          max_tokens: 32768
          overflow: reject
`
		}
		t.Run(name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes([]byte(raw))
			if err != nil {
				t.Fatal(err)
			}
			plan, err := CompileModelBindings(cfg)
			if err != nil {
				t.Fatal(err)
			}
			bound, ok := plan.Lookup("care", "classifier.content-risk")
			if !ok {
				t.Fatal("named classifier did not resolve default Hazard deployment")
			}
			point := bound.Binding.OperatingPoint
			if point == nil || point.Path != "operating_point.json" || point.SHA256 != "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5" {
				t.Fatalf("operating-point identity lost: %+v", point)
			}
			if bound.Binding.Contract != RemoteClassifierContractLabelScores || bound.Binding.Adapter != "modernbert" {
				t.Fatalf("independent-score semantics changed: %+v", bound.Binding)
			}
			if _, leaked := plan.Lookup(DefaultRecipeName, "classifier.content-risk"); leaked {
				t.Fatal("named Hazard binding leaked into the default API scope")
			}
			if cfg.SafetyModels.Hazard.ModelID != "" {
				t.Fatal("generic Hazard binding enabled implicit scalar cascade")
			}
			if amd && (bound.Deployment.Provider != "ort" || bound.Deployment.Device != "migraphx:0" || bound.Deployment.Precision != "native" || bound.Deployment.CompilationCacheDir != "/tmp/router-compilation-cache") {
				t.Fatalf("explicit AMD deployment replaced by CPU defaults: %+v", bound.Deployment)
			}
			if bound.Deployment.Input != (ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}) {
				t.Fatalf("complete-document admission lost: %+v", bound.Deployment.Input)
			}
			data, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
			if err != nil {
				t.Fatal(err)
			}
			roundTrip, err := ParseYAMLBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			roundPlan, err := CompileModelBindings(roundTrip)
			if err != nil {
				t.Fatal(err)
			}
			roundBound, ok := roundPlan.Lookup("care", "classifier.content-risk")
			if !ok || !reflect.DeepEqual(bound, roundBound) {
				t.Fatalf("binding changed on canonical round trip: before=%+v after=%+v", bound, roundBound)
			}
		})
	}
}

func TestDefaultHazardOverrideReplacesWholeEntry(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
global:
  model_catalog:
    deployments:
      hazard:
        provider: candle
        artifact: models/operator-hazard
`))
	if err != nil {
		t.Fatal(err)
	}
	deployment := cfg.ModelDeployments["hazard"]
	if deployment.Artifact != "models/operator-hazard" || deployment.Revision != "" || deployment.Input.MaxTokens != 0 {
		t.Fatalf("operator deployment inherited unrelated default fields: %+v", deployment)
	}
}
