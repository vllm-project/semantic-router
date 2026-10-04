//go:build !windows && (amd64 || arm64)

package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStandaloneCanonicalStartupRejectsInvalidBindings(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.MoMRegistry = map[string]string{"models/lora_model": "unrelated/discoverable-model"}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "missing", Adapter: "mmbert", Contract: "embedding.v1"}}
	service, err := NewClassificationServiceFromConfig(cfg)
	if err == nil || service != nil {
		if service != nil {
			_ = service.Close()
		}
		t.Fatal("invalid canonical config fell back to an autodiscovered/placeholder service")
	}
	idle, err := NewClassificationServiceWithAutoDiscovery(&config.RouterConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer idle.Close()
	if bindings, ok := idle.PreparedBindings(); !ok || len(bindings) != 0 {
		t.Fatalf("empty canonical config discovered models: %+v", bindings)
	}
}
