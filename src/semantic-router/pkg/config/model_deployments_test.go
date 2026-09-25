package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func testDeploymentConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"shared-encoder": {Artifact: "models/maintained-encoder", Revision: "frozen-revision", Provider: "candle", Device: "cpu", Precision: "native", Input: ModelInputBudget{MaxTokens: 512, Overflow: "reject"}},
		"other-encoder":  {Artifact: "models/other-maintained-encoder", Provider: "ort", Device: "migraphx:1", Precision: "native"},
	}
	cfg.ModelAdmission = map[string]AdmissionConfig{"shared-encoder": {MaxConcurrency: 2, MaxQueue: 3, OnOverflow: "shed"}}
	cfg.Recipes = []RoutingRecipe{
		{Name: "support", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			"domain_classifier": {Deployment: "shared-encoder", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert", Head: "domain-head", MappingPath: "domain-labels.json"},
			"pii_classifier":    {Deployment: "shared-encoder", Contract: RemoteClassifierContractTokenSpans, Adapter: "modernbert", Head: "pii-head", MappingPath: "pii-labels.json"},
		}}},
		{Name: "coding", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			"domain_classifier": {Deployment: "other-encoder", Contract: RemoteClassifierContractLabelDistribution, Adapter: "mmbert32k"},
		}}},
	}
	return cfg
}

func TestCompileModelBindingsPreservesResourceAndTaskIdentity(t *testing.T) {
	cfg := testDeploymentConfig()
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	domain, ok := plan.Lookup("support", "domain_classifier")
	if !ok || domain.Binding.Head != "domain-head" || domain.Admission.MaxConcurrency != 2 {
		t.Fatalf("domain=%+v", domain)
	}
	pii, ok := plan.Lookup("support", "pii_classifier")
	if !ok || pii.Deployment != domain.Deployment || pii.Binding.Head == domain.Binding.Head {
		t.Fatalf("head/resource identities conflated: %+v", pii)
	}
	if _, ok := plan.Lookup("coding", "pii_classifier"); ok {
		t.Fatal("foreign recipe lookup succeeded")
	}
	coding, _ := plan.Lookup("coding", "domain_classifier")
	if coding.Deployment.Device != "migraphx:1" {
		t.Fatal("wrong recipe deployment")
	}
	cfg.ModelDeployments["shared-encoder"] = ModelDeployment{}
	retained, _ := plan.Lookup("support", "domain_classifier")
	if retained.Deployment.Artifact == "" {
		t.Fatal("compiled plan aliases mutable configuration")
	}
}

func TestCompileModelBindingsRejectsInvalidPreparation(t *testing.T) {
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
		{"wrong device", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Device = "migraphx:0"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "incompatible"},
		{"negative budget", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Input.MaxTokens = -1
			cfg.ModelDeployments["shared-encoder"] = d
		}, "must not be negative"},
		{"wrong contract", func(cfg *RouterConfig) {
			b := cfg.Recipes[0].Profile.ModelBindings["domain_classifier"]
			b.Contract = RemoteClassifierContractScore
			cfg.Recipes[0].Profile.ModelBindings["domain_classifier"] = b
		}, "contract must"},
		{"missing adapter", func(cfg *RouterConfig) {
			b := cfg.Recipes[0].Profile.ModelBindings["domain_classifier"]
			b.Adapter = ""
			cfg.Recipes[0].Profile.ModelBindings["domain_classifier"] = b
		}, "adapter is required"},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := testDeploymentConfig()
			test.mutate(cfg)
			_, err := CompileModelBindings(cfg)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want %q", err, test.want)
			}
		})
	}
}

func TestNamedDeploymentAdmissionAndCanonicalRoundTrip(t *testing.T) {
	cfg := testDeploymentConfig()
	deployment := cfg.ModelDeployments["other-encoder"]
	deployment.CompilationCacheDir = "/var/cache/semantic-router/migraphx"
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
	if roundTrip.ModelDeployments["other-encoder"] != cfg.ModelDeployments["other-encoder"] {
		t.Fatal("deployment lost during canonical round trip")
	}
	recipes := canonicalRecipesFromRouterConfig(cfg)
	if len(recipes) != 2 || recipes[0].Routing.ModelBindings["pii_classifier"].Head != "pii-head" {
		t.Fatal("recipe binding lost during export")
	}
	scoped := cfg.ConfigForRecipe(&cfg.Recipes[0])
	if scoped.ModelBindings["domain_classifier"].Deployment != "shared-encoder" {
		t.Fatal("recipe view lost bindings")
	}
}

func TestRemovedClassifierSessionBankOptionIsRejected(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
global:
  model_catalog:
    deployments:
      classifier:
        artifact: models/classifier
        provider: ort
        device: migraphx:0
        input:
          max_tokens: 8192
          overflow: reject
        short_sequence_tokens: 512
`))
	if err == nil || !strings.Contains(err.Error(), `unknown field "short_sequence_tokens"`) {
		t.Fatalf("removed session-bank option must be rejected, got %v", err)
	}
}

func TestCompilationCacheRequiresExplicitMIGraphXPlacement(t *testing.T) {
	for _, test := range []struct {
		name, provider, device, directory string
		valid                             bool
	}{
		{"default off", "ort", "cpu", "", true},
		{"MIGraphX", "ort", "migraphx:2", "/var/cache/semantic-router/migraphx", true},
		{"CPU", "ort", "cpu", "/cache", false},
		{"ROCm", "ort", "rocm:0", "/cache", false},
		{"Candle", "candle", "cpu", "/cache", false},
		{"HTTP", "http", "", "/cache", false},
		{"relative", "ort", "migraphx:0", "cache", false},
		{"spaces", "ort", "migraphx:0", " /cache", false},
		{"null", "ort", "migraphx:0", "/cache\x00", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := testDeploymentConfig()
			d := cfg.ModelDeployments["other-encoder"]
			d.Provider, d.Device, d.CompilationCacheDir = test.provider, test.device, test.directory
			if err := d.ValidateCompilationCache(); (err == nil) != test.valid {
				t.Fatalf("cache validation: %v", err)
			}
			if test.provider != "http" {
				cfg.ModelDeployments["other-encoder"] = d
				if _, err := CompileModelBindings(cfg); (err == nil) != test.valid {
					t.Fatalf("compiled deployment: %v", err)
				}
			}
		})
	}
}

func TestDormantComplexityRejectsUnsupportedLocalProvider(t *testing.T) {
	for _, provider := range []string{"candle", "ort"} {
		t.Run(provider, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.ModelDeployments = map[string]ModelDeployment{"local": {Provider: provider, Artifact: "/mounted/local"}}
			cfg.Recipes = []RoutingRecipe{{Name: "dormant", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{"complexity": {Deployment: "local", Contract: "score.v1", Adapter: "mmbert"}}}}}
			if _, err := CompileModelBindings(cfg); err == nil {
				t.Fatal("dormant recipe accepted unsupported local complexity")
			}
		})
	}
}
