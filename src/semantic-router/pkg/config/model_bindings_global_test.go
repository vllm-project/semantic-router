package config

import (
	"strings"
	"testing"
)

func TestGlobalModelBindingDeclarationsValidateBeforeProvisioning(t *testing.T) {
	cfg := testDeploymentConfig()
	cfg.GlobalModelBindings = map[string]ModelBinding{SignalTypePII: {Deployment: "missing"}}
	if _, err := CompileModelBindings(cfg); err == nil || !strings.Contains(err.Error(), "global.model_catalog.bindings.pii") {
		t.Fatalf("global declaration error = %v", err)
	}
}
