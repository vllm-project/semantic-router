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
