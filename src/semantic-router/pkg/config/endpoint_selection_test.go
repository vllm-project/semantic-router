package config_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSelectEndpointForConversation(t *testing.T) {
	// Create a test config with multiple endpoints
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "endpoint1",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o"},
				Weight:  1,
			},
			{
				Name:    "endpoint2",
				Address: "127.0.0.2",
				Port:    8000,
				Models:  []string{"gpt-4o"},
				Weight:  1,
			},
			{
				Name:    "endpoint3",
				Address: "127.0.0.3",
				Port:    8000,
				Models:  []string{"gpt-4o"},
				Weight:  1,
			},
		},
	}

	// Test that the same conversation ID always maps to the same endpoint
	conversationID1 := "resp_abc123"
	conversationID2 := "resp_xyz789"

	endpoint1_1, found1_1 := cfg.SelectEndpointForConversation("gpt-4o", conversationID1)
	assert.True(t, found1_1, "Should find endpoint for conversation 1")

	endpoint1_2, found1_2 := cfg.SelectEndpointForConversation("gpt-4o", conversationID1)
	assert.True(t, found1_2, "Should find endpoint for conversation 1 again")

	// Same conversation ID should always map to the same endpoint
	assert.Equal(t, endpoint1_1, endpoint1_2, "Same conversation should map to same endpoint")

	endpoint2_1, found2_1 := cfg.SelectEndpointForConversation("gpt-4o", conversationID2)
	assert.True(t, found2_1, "Should find endpoint for conversation 2")

	endpoint2_2, found2_2 := cfg.SelectEndpointForConversation("gpt-4o", conversationID2)
	assert.True(t, found2_2, "Should find endpoint for conversation 2 again")

	// Same conversation ID should always map to the same endpoint
	assert.Equal(t, endpoint2_1, endpoint2_2, "Same conversation should map to same endpoint")

	// All selected endpoints should be valid
	validEndpoints := []string{"127.0.0.1:8000", "127.0.0.2:8000", "127.0.0.3:8000"}
	assert.Contains(t, validEndpoints, endpoint1_1, "Selected endpoint should be valid")
	assert.Contains(t, validEndpoints, endpoint2_1, "Selected endpoint should be valid")
}

func TestSelectEndpointForConversation_SingleEndpoint(t *testing.T) {
	// Create a config with only one endpoint
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "endpoint1",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o"},
				Weight:  1,
			},
		},
	}

	// With a single endpoint, it should always return that endpoint
	endpoint, found := cfg.SelectEndpointForConversation("gpt-4o", "resp_abc123")
	assert.True(t, found, "Should find endpoint")
	assert.Equal(t, "127.0.0.1:8000", endpoint, "Should return the only endpoint")
}

func TestSelectEndpointForConversation_NoEndpoints(t *testing.T) {
	// Create a config with no endpoints for the model
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "endpoint1",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"other-model"},
				Weight:  1,
			},
		},
	}

	// Should not find an endpoint for a model that doesn't exist
	endpoint, found := cfg.SelectEndpointForConversation("gpt-4o", "resp_abc123")
	assert.False(t, found, "Should not find endpoint for non-existent model")
	assert.Empty(t, endpoint, "Endpoint should be empty")
}
