package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ImageGenPluginConfig represents configuration for image generation plugin
type ImageGenPluginConfig struct {
	// Enable image generation for this decision
	Enabled bool `json:"enabled" yaml:"enabled"`

	// Backend type: "vllm_omni", "openai", "replicate"
	Backend string `json:"backend" yaml:"backend"`

	// Backend-specific configuration
	BackendConfig interface{} `json:"backend_config,omitempty" yaml:"backend_config,omitempty"`

	// Default image parameters
	DefaultWidth  int `json:"default_width,omitempty" yaml:"default_width,omitempty"`
	DefaultHeight int `json:"default_height,omitempty" yaml:"default_height,omitempty"`

	// Maximum inference steps (for diffusion models)
	MaxInferenceSteps int `json:"max_inference_steps,omitempty" yaml:"max_inference_steps,omitempty"`

	// Timeout in seconds
	TimeoutSeconds int `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
}

// VLLMOmniImageGenConfig represents configuration for vLLM-Omni image generation
type VLLMOmniImageGenConfig struct {
	// Base URL for vLLM-Omni server (e.g., "http://localhost:8001")
	BaseURL string `json:"base_url" yaml:"base_url"`

	// Model name to use (e.g., "Qwen/Qwen-Image")
	Model string `json:"model,omitempty" yaml:"model,omitempty"`

	// Default number of inference steps
	NumInferenceSteps int `json:"num_inference_steps,omitempty" yaml:"num_inference_steps,omitempty"`

	// CFG scale for guidance
	CFGScale float64 `json:"cfg_scale,omitempty" yaml:"cfg_scale,omitempty"`

	// Seed for reproducibility (optional)
	Seed *int `json:"seed,omitempty" yaml:"seed,omitempty"`
}

// OpenAIImageGenConfig represents configuration for OpenAI image generation
type OpenAIImageGenConfig struct {
	// OpenAI API key
	APIKey string `json:"api_key" yaml:"api_key"`

	// Base URL (defaults to https://api.openai.com/v1)
	BaseURL string `json:"base_url,omitempty" yaml:"base_url,omitempty"`

	// Model to use (e.g., "gpt-image-1", "dall-e-3")
	Model string `json:"model,omitempty" yaml:"model,omitempty"`

	// Image quality: "standard" or "hd"
	Quality string `json:"quality,omitempty" yaml:"quality,omitempty"`

	// Style: "vivid" or "natural"
	Style string `json:"style,omitempty" yaml:"style,omitempty"`
}

// GetImageGenConfig returns the image generation plugin configuration for a decision
func (d *Decision) GetImageGenConfig() *ImageGenPluginConfig {
	pluginConfig := d.GetPluginConfig("image_gen")
	if pluginConfig == nil {
		return nil
	}

	result := &ImageGenPluginConfig{}
	if err := unmarshalPluginConfig(pluginConfig, result); err != nil {
		logging.Errorf("Failed to unmarshal image_gen config: %v", err)
		return nil
	}

	// Unmarshal backend-specific config based on Backend type
	if result.BackendConfig != nil && result.Backend != "" {
		var backendConfig interface{}
		switch result.Backend {
		case "vllm_omni":
			backendConfig = &VLLMOmniImageGenConfig{}
		case "openai":
			backendConfig = &OpenAIImageGenConfig{}
		default:
			logging.Warnf("Unknown image_gen backend type: %s", result.Backend)
			return result
		}

		if err := unmarshalPluginConfig(result.BackendConfig, backendConfig); err != nil {
			logging.Errorf("Failed to unmarshal image_gen backend config for %s: %v", result.Backend, err)
		} else {
			result.BackendConfig = backendConfig
		}
	}

	return result
}

// Validate validates the image generation plugin configuration
func (c *ImageGenPluginConfig) Validate() error {
	if !c.Enabled {
		return nil
	}

	if c.Backend == "" {
		return fmt.Errorf("image_gen backend is required when enabled")
	}

	switch c.Backend {
	case "vllm_omni":
		if c.BackendConfig == nil {
			return fmt.Errorf("backend_config is required for vllm_omni backend")
		}
		vllmConfig, ok := c.BackendConfig.(*VLLMOmniImageGenConfig)
		if !ok {
			return fmt.Errorf("backend_config must be VLLMOmniImageGenConfig for vllm_omni backend")
		}
		if vllmConfig.BaseURL == "" {
			return fmt.Errorf("base_url is required for vllm_omni backend")
		}
	case "openai":
		if c.BackendConfig == nil {
			return fmt.Errorf("backend_config is required for openai backend")
		}
		openaiConfig, ok := c.BackendConfig.(*OpenAIImageGenConfig)
		if !ok {
			return fmt.Errorf("backend_config must be OpenAIImageGenConfig for openai backend")
		}
		if openaiConfig.APIKey == "" {
			return fmt.Errorf("api_key is required for openai backend")
		}
	default:
		return fmt.Errorf("unknown image_gen backend: %s", c.Backend)
	}

	return nil
}
