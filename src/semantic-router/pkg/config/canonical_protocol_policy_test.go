package config

import (
	"strings"
	"testing"
)

func protocolPolicyYAML(lossy string) string {
	doc := `
version: v0.3
providers:
  defaults:
    model: private-model
routing:
  modelCards:
    - name: private-model
global:
  router:
    strategy: priority
`
	if lossy == "" {
		return doc
	}
	return strings.Replace(doc, "  router:\n", "  router:\n    protocol:\n      lossy_features: "+lossy+"\n", 1)
}

func TestProtocolLossyFeaturesLayering(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(protocolPolicyYAML("")))
	if err != nil {
		t.Fatalf("parse default doc: %v", err)
	}
	if cfg.ProtocolLossyFeatures != "" {
		t.Fatalf("omitted knob must stay empty, got %q", cfg.ProtocolLossyFeatures)
	}

	cfg, err = ParseYAMLBytes([]byte(protocolPolicyYAML(ProtocolLossyAllowWithDiagnostic)))
	if err != nil {
		t.Fatalf("parse permissive doc: %v", err)
	}
	if cfg.ProtocolLossyFeatures != ProtocolLossyAllowWithDiagnostic {
		t.Fatalf("permissive knob not layered, got %q", cfg.ProtocolLossyFeatures)
	}

	cfg, err = ParseYAMLBytes([]byte(protocolPolicyYAML(" " + ProtocolLossyReject + " ")))
	if err != nil {
		t.Fatalf("parse reject doc: %v", err)
	}
	if cfg.ProtocolLossyFeatures != ProtocolLossyReject {
		t.Fatalf("whitespace must be trimmed, got %q", cfg.ProtocolLossyFeatures)
	}
}

func TestProtocolLossyFeaturesValidation(t *testing.T) {
	if err := validateProtocolTranslationPolicy(nil); err != nil {
		t.Fatalf("nil config must pass: %v", err)
	}
	for _, value := range []string{"", ProtocolLossyReject, ProtocolLossyAllowWithDiagnostic} {
		cfg := &RouterConfig{ProtocolLossyFeatures: value}
		if err := validateProtocolTranslationPolicy(cfg); err != nil {
			t.Fatalf("value %q must pass: %v", value, err)
		}
	}
	cfg := &RouterConfig{ProtocolLossyFeatures: "allow"}
	err := validateProtocolTranslationPolicy(cfg)
	if err == nil || !strings.Contains(err.Error(), "global.router.protocol.lossy_features") {
		t.Fatalf("unknown value must be rejected with the config path, got: %v", err)
	}
}
