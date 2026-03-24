package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMemoryOptOut_DisabledRoutes(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{
				Enabled: true,
				DisabledRoutes: []string{
					"/v1/chat/completions",
					"/v1/completions",
				},
			},
		},
	}

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{"chat completions disabled", "/v1/chat/completions", true},
		{"completions disabled", "/v1/completions", true},
		{"responses enabled", "/v1/responses", false},
		{"unknown route enabled", "/v1/embeddings", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := router.isMemoryDisabledForRoute(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMemoryOptOut_DisabledModels(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{
				Enabled: true,
				DisabledModels: []string{
					"gpt-4-with-mem0",
					"claude-3-langmem",
				},
			},
		},
	}

	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{"gpt-4-with-mem0 disabled", "gpt-4-with-mem0", true},
		{"claude-3-langmem disabled", "claude-3-langmem", true},
		{"gpt-4 enabled", "gpt-4", false},
		{"claude-3-sonnet enabled", "claude-3-sonnet", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := router.isMemoryDisabledForModel(tt.model)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMemoryOptOut_NoDisabledRoutes(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{
				Enabled:        true,
				DisabledRoutes: nil,
			},
		},
	}

	result := router.isMemoryDisabledForRoute("/v1/chat/completions")
	assert.False(t, result, "should return false when no routes are disabled")
}

func TestMemoryOptOut_NoDisabledModels(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{
				Enabled:        true,
				DisabledModels: []string{},
			},
		},
	}

	result := router.isMemoryDisabledForModel("gpt-4")
	assert.False(t, result, "should return false when no models are disabled")
}
