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
	globalConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "global_route"}},
		},
	}
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

	restoreGlobalConfig := replaceGlobalConfigForServiceTest(globalConfig)
	t.Cleanup(restoreGlobalConfig)

	service := NewClassificationService(nil, oldConfig)
	service.RefreshRuntimeConfig(newConfig)

	if got := service.GetConfig(); got != newConfig {
		t.Fatalf("service.GetConfig() = %p, want %p", got, newConfig)
	}
	if got := config.Get(); got != globalConfig {
		t.Fatalf("config.Get() = %p, want unchanged global config %p", got, globalConfig)
	}
}

func replaceGlobalConfigForServiceTest(newCfg *config.RouterConfig) func() {
	previous := config.Get()
	config.Replace(newCfg)
	return func() {
		if previous != nil {
			config.Replace(previous)
			return
		}
		config.Replace(&config.RouterConfig{})
	}
}
