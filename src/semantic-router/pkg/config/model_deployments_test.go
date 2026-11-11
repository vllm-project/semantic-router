package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func testDeploymentConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"shared-encoder": {Artifact: "models/maintained-encoder", Revision: strings.Repeat("f", 40), Provider: ModelRuntimeProvider, Device: "cpu", Input: ModelInputBudget{MaxTokens: 512, Overflow: "reject"}},
		"other-encoder":  {Artifact: "models/other-maintained-encoder", Provider: ModelRuntimeProvider, Device: "rocm:1"},
	}
	cfg.ModelAdmission = map[string]AdmissionConfig{"shared-encoder": {MaxConcurrency: 2, MaxQueue: 3, OnOverflow: "shed"}}
	cfg.Recipes = []RoutingRecipe{
		{Name: "support", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			SignalTypeJailbreak: {Deployment: "shared-encoder"},
			SignalTypePII:       {Deployment: "shared-encoder"},
		}}},
		{Name: "coding", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			SignalTypeJailbreak: {Deployment: "other-encoder"},
		}}},
	}
	return cfg
}

func TestRecipeBindingsResolveInTheirOwnRecipe(t *testing.T) {
	cfg := testDeploymentConfig()
	if err := ValidateModelBindings(cfg); err != nil {
		t.Fatal(err)
	}
	support, coding := cfg.ConfigForRecipe(&cfg.Recipes[0]), cfg.ConfigForRecipe(&cfg.Recipes[1])
	guard, ok, err := support.ResolveTask(SignalTypeJailbreak)
	if err != nil || !ok || guard.Name != "shared-encoder" || guard.Source != ResolvedByRecipeBinding {
		t.Fatalf("guard=%+v %v", guard, err)
	}
	pii, ok, err := support.ResolveTask(SignalTypePII)
	if err != nil || !ok || !reflect.DeepEqual(pii.Deployment, guard.Deployment) || pii.Consumer != SignalTypePII {
		t.Fatalf("task/resource identities conflated: %+v %v", pii, err)
	}
	if _, ok, _ := coding.ResolveTask(SignalTypePII); ok {
		t.Fatal("a recipe resolved another recipe's binding")
	}
	if other, _, _ := coding.ResolveTask(SignalTypeJailbreak); other.Deployment.Device != "rocm:1" {
		t.Fatal("wrong recipe deployment")
	}
}

func TestModelBindingsRejectInvalidPreparation(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*RouterConfig)
		want   string
	}{
		{"unknown deployment", func(cfg *RouterConfig) { delete(cfg.ModelDeployments, "shared-encoder") }, "unknown deployment"},
		{"foreign provider", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Provider = "unknown"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "unsupported provider"},
		{"malformed device", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Device = "CUDA 0"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "device must be"},
		{"no rows per forward", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.MaxNumSeqs = intPtr(0)
			cfg.ModelDeployments["shared-encoder"] = d
		}, "max_num_seqs must be at least 1"},
		{"no input tokens", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.MaxModelLen = intPtr(0)
			cfg.ModelDeployments["shared-encoder"] = d
		}, "max_model_len must be at least 1"},
		{"input limit on an attached endpoint", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.MaxModelLen, d.Device, d.Endpoint = intPtr(32000), "", "http://runtime:8000"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "max_model_len applies only to a managed worker"},
		{"input limit on replicas", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.MaxModelLen, d.Device, d.Replicas = intPtr(32000), "", []ModelReplica{{Device: "cpu"}, {Device: "cpu"}}
			cfg.ModelDeployments["shared-encoder"] = d
		}, "max_model_len applies only to a managed worker"},
		{"unknown dtype", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Dtype = "half"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "dtype must be one of auto, float32, bfloat16 or float16"},
		{"negative budget", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Input.MaxTokens = -1
			cfg.ModelDeployments["shared-encoder"] = d
		}, "must not be negative"},
		{"a binding of no signal type", func(cfg *RouterConfig) {
			cfg.Recipes[0].Profile.ModelBindings["prompt_guard"] = ModelBinding{Deployment: "shared-encoder"}
		}, `unknown signal type "prompt_guard"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := testDeploymentConfig()
			test.mutate(cfg)
			err := ValidateModelBindings(cfg)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want %q", err, test.want)
			}
		})
	}
}

func TestPluginAcceleratorNamesAreTheRuntimes(t *testing.T) {
	cfg := testDeploymentConfig()
	d := cfg.ModelDeployments["shared-encoder"]
	d.Device = "npu:1"
	cfg.ModelDeployments["shared-encoder"] = d
	if err := ValidateModelBindings(cfg); err != nil {
		t.Fatalf("a well-formed plugin accelerator compiles; the runtime decides whether it has it: %v", err)
	}
}

func TestEngineArgumentsCompileOnModelRuntimeDeployments(t *testing.T) {
	cfg := testDeploymentConfig()
	d := cfg.ModelDeployments["shared-encoder"]
	window, shared := 0.5, false
	d.MaxNumSeqs, d.MaxNumBatchedTokens, d.BatchWindowMS = intPtr(1), intPtr(4096), &window
	d.Dtype, d.EnableSharedContext = "bfloat16", &shared
	d.MaxModelLen = intPtr(32000)
	cfg.ModelDeployments["shared-encoder"] = d
	if err := ValidateModelBindings(cfg); err != nil {
		t.Fatalf("engine arguments within their ranges compile: %v", err)
	}
	if !d.HasEngineArgs() || cfg.ModelDeployments["shared-encoder"].WithDefaults().Dtype != "bfloat16" {
		t.Fatalf("defaults must keep the declared engine arguments: %+v", d)
	}
}

func TestNamedDeploymentAdmissionAndCanonicalRoundTrip(t *testing.T) {
	cfg := testDeploymentConfig()
	deployment := cfg.ModelDeployments["other-encoder"]
	deployment.Input.MaxTokens = 8192
	cfg.ModelDeployments["other-encoder"] = deployment
	if err := validateModelAdmissionContracts(cfg); err != nil {
		t.Fatal(err)
	}
	global := canonicalModelCatalogFromRouterConfig(cfg)
	data, err := yaml.Marshal(global)
	if err != nil {
		t.Fatal(err)
	}
	var decoded CanonicalModelCatalog
	if err := yaml.UnmarshalStrict(data, &decoded); err != nil {
		t.Fatal(err)
	}
	var roundTrip RouterConfig
	applyCanonicalModelCatalogGlobal(&roundTrip, decoded)
	if !reflect.DeepEqual(roundTrip.ModelDeployments["other-encoder"], cfg.ModelDeployments["other-encoder"]) {
		t.Fatal("deployment lost during canonical round trip")
	}
	recipes := canonicalRecipesFromRouterConfig(cfg)
	if len(recipes) != 2 || recipes[0].ModelBindings[SignalTypePII] != (ModelBinding{Deployment: "shared-encoder"}) {
		t.Fatal("recipe binding lost during export")
	}
	scoped := cfg.ConfigForRecipe(&cfg.Recipes[0])
	if scoped.ModelBindings[SignalTypeJailbreak].Deployment != "shared-encoder" {
		t.Fatal("recipe view lost bindings")
	}
}

func TestRemovedClassifierSessionBankOptionIsRejected(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.5
global:
  model_catalog:
    deployments:
      classifier:
        artifact: /models/classifier
        provider: model_runtime
        device: rocm:0
        input:
          max_tokens: 8192
          overflow: reject
        short_sequence_tokens: 512
`))
	if err == nil || !strings.Contains(err.Error(), `unknown field "short_sequence_tokens"`) {
		t.Fatalf("removed session-bank option must be rejected, got %v", err)
	}
}

func TestABindingToAChatModelIsRefused(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{"chat": {Provider: OpenAIChatProvider, Endpoint: "http://chat:8000/v1", Model: "chat"}}
	cfg.Recipes = []RoutingRecipe{{Name: "dormant", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{SignalTypeComplexity: {Deployment: "chat"}}}}}
	if err := ValidateModelBindings(cfg); err == nil || !strings.Contains(err.Error(), "answers no signal") {
		t.Fatalf("a complexity binding accepted a chat model: %v", err)
	}
}
