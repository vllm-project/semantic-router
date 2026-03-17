package main

import (
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyKubernetesConfigUpdateEnsuresModelsBeforeReplace(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	cfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	order := make([]string, 0, 2)

	ensureKubernetesConfigModels = func(got *config.RouterConfig) error {
		order = append(order, "ensure")
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return nil
	}
	replaceKubernetesRuntimeConfig = func(got *config.RouterConfig) {
		order = append(order, "replace")
		if got != cfg {
			t.Fatalf("replaceKubernetesRuntimeConfig() cfg = %p, want %p", got, cfg)
		}
	}

	if err := applyKubernetesConfigUpdate(cfg); err != nil {
		t.Fatalf("applyKubernetesConfigUpdate() error = %v", err)
	}

	wantOrder := []string{"ensure", "replace"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("applyKubernetesConfigUpdate() order = %v, want %v", order, wantOrder)
	}
}

func TestApplyKubernetesConfigUpdateSkipsReplaceOnEnsureFailure(t *testing.T) {
	restoreKubernetesUpdateSeams := stubKubernetesUpdateSeams(t)
	defer restoreKubernetesUpdateSeams()

	cfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	ensureKubernetesConfigModels = func(got *config.RouterConfig) error {
		if got != cfg {
			t.Fatalf("ensureKubernetesConfigModels() cfg = %p, want %p", got, cfg)
		}
		return errors.New("download failed")
	}
	replaceKubernetesRuntimeConfig = func(got *config.RouterConfig) {
		t.Fatalf("replaceKubernetesRuntimeConfig() should not be called on ensure failure")
	}

	err := applyKubernetesConfigUpdate(cfg)
	if err == nil {
		t.Fatal("applyKubernetesConfigUpdate() error = nil, want failure")
	}
	if got := err.Error(); got != "failed to ensure models for kubernetes config update: download failed" {
		t.Fatalf("applyKubernetesConfigUpdate() error = %q", got)
	}
}

func stubKubernetesUpdateSeams(t *testing.T) func() {
	t.Helper()

	originalEnsure := ensureKubernetesConfigModels
	originalReplace := replaceKubernetesRuntimeConfig

	return func() {
		ensureKubernetesConfigModels = originalEnsure
		replaceKubernetesRuntimeConfig = originalReplace
	}
}
