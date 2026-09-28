package config

import "testing"

func TestContextCompressionCurrentUserOptInValidation(t *testing.T) {
	for _, mode := range []string{"", "preserve", "truncate", "extractive", "recoverable", "drop"} {
		cfg := &ContextCompressionPluginConfig{Enabled: true, Targets: &ContextCompressionTargetsConfig{CurrentUser: ContextCompressionCurrentUserConfig{Mode: mode}}}
		err := validateContextCompressionTargets(cfg, "test")
		valid := mode == "" || mode == "preserve" || mode == "truncate"
		if (err == nil) != valid {
			t.Fatalf("mode=%q err=%v", mode, err)
		}
	}
	cfg := &ContextCompressionPluginConfig{Enabled: true, Targets: &ContextCompressionTargetsConfig{History: ContextCompressionTargetConfig{Mode: "truncate"}}}
	if err := validateContextCompressionTargets(cfg, "test"); err == nil {
		t.Fatal("live user permission leaked to other target")
	}
}
