package config

import "testing"

func TestPromptGuardWindowContract(t *testing.T) {
	base := PromptGuardConfig{
		Variant: PromptGuardVariantMmBERT32K, MaxSequenceLength: 32768,
		Window: &SequenceHeadWindowConfig{Size: 128, Overlap: 63},
	}
	if err := validatePromptGuardBackendConfig(&base); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*PromptGuardConfig){
		func(c *PromptGuardConfig) { c.Backend = &RemoteClassifierBackend{}; c.Variant = "" },
		func(c *PromptGuardConfig) { c.Protocol = PromptGuardProtocolHTTPChat; c.Variant = "" },
		func(c *PromptGuardConfig) { c.Variant = PromptGuardVariantCandle },
		func(c *PromptGuardConfig) { c.Window = &SequenceHeadWindowConfig{Size: 0} },
		func(c *PromptGuardConfig) { c.Window = &SequenceHeadWindowConfig{Size: 128, Overlap: 128} },
		func(c *PromptGuardConfig) { c.MaxSequenceLength = 64 },
		func(c *PromptGuardConfig) { c.PositiveLabels = []string{"jailbreak", "jailbreak"} },
	} {
		cfg := base
		mutate(&cfg)
		if err := validatePromptGuardBackendConfig(&cfg); err == nil {
			t.Fatalf("invalid window accepted: %+v", cfg)
		}
	}
	legacy := PromptGuardConfig{Variant: PromptGuardVariantCandle}
	if err := validatePromptGuardBackendConfig(&legacy); err != nil {
		t.Fatal(err)
	}
}

func TestPromptGuardWindowAcceptsExplicitLocalBindingWithoutLegacyVariant(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.PromptGuard.Window = &SequenceHeadWindowConfig{Size: 1024, Overlap: 128}
	cfg.ModelDeployments = map[string]ModelDeployment{"guard": {Provider: "candle", Artifact: "owned", Input: ModelInputBudget{MaxTokens: 2048}}}
	cfg.ModelBindings = map[string]ModelBinding{"prompt_guard": {Deployment: "guard", Adapter: "modernbert", Contract: RemoteClassifierContractLabelDistribution}}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := ProjectRecipeModelBindings(cfg, plan, DefaultRecipeName)
	if err != nil {
		t.Fatal(err)
	}
	if err := validatePromptGuardBackend(projected); err != nil {
		t.Fatal(err)
	}
	d := cfg.ModelDeployments["guard"]
	d.Input.MaxTokens = 512
	cfg.ModelDeployments["guard"] = d
	if err := validatePromptGuardBackend(cfg); err == nil {
		t.Fatal("ignored actual bound budget")
	}
	d.Provider = "http"
	d.Input.MaxTokens = 0
	cfg.ModelDeployments["guard"] = d
	if err := validatePromptGuardBackend(cfg); err == nil {
		t.Fatal("accepted remote token windows")
	}
}
