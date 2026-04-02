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
