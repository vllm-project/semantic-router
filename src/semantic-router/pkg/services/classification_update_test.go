package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassificationServiceRefreshRuntimeConfigRefreshesClassifierConfig(t *testing.T) {
	oldConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "old_route"}},
		},
	}
	newConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "new_route"}},
		},
	}

	service := &ClassificationService{
		classifier: &classification.Classifier{Config: oldConfig},
		config:     oldConfig,
	}

	service.RefreshRuntimeConfig(newConfig)

	if service.config != newConfig {
		t.Fatalf("expected service config to be updated")
	}
	if service.classifier == nil {
		t.Fatalf("expected classifier to remain available")
	}
	if service.classifier.Config != newConfig {
		t.Fatalf("expected classifier config to be updated")
	}
}

func TestClassificationServiceRefreshRuntimeConfigDoesNotReplaceGlobalConfig(t *testing.T) {
	previousCfg := config.Get()
	stableCfg := previousCfg
	if stableCfg == nil {
		stableCfg = &config.RouterConfig{}
	}
	config.Replace(stableCfg)
	t.Cleanup(func() {
		config.Replace(stableCfg)
		if previousCfg != nil {
			config.Replace(previousCfg)
		}
	})

	service := &ClassificationService{
		classifier: &classification.Classifier{Config: stableCfg},
		config:     stableCfg,
	}
	updatedCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "service-local"},
	}

	service.RefreshRuntimeConfig(updatedCfg)

	if got := config.Get(); got != stableCfg {
		t.Fatalf("config.Get() = %p, want unchanged %p", got, stableCfg)
	}
	if service.config != updatedCfg {
		t.Fatalf("service.config = %p, want %p", service.config, updatedCfg)
	}
}
