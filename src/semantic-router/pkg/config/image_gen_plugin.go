package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ModalityDetectionMethod defines how modality is detected
const (
	// ModalityDetectionClassifier uses the mmBERT-32K ML classifier (3-class: AR/DIFFUSION/BOTH)
	ModalityDetectionClassifier = "classifier"
	// ModalityDetectionKeyword uses keyword pattern matching (2-class: AR/DIFFUSION)
	ModalityDetectionKeyword = "keyword"
	// ModalityDetectionHybrid uses classifier first, keyword as fallback/confirmation (default)
	ModalityDetectionHybrid = "hybrid"
)

// ModalityRoutingConfig is the top-level configuration for modality-based routing.
// It runs before the decision engine and classifies every request as AR / DIFFUSION / BOTH.
//
//   - AR        → passthrough to the AR text model (normal flow)
//   - DIFFUSION → short-circuit: generate image via diffusion endpoint, return immediately
//   - BOTH      → short-circuit: call AR for text AND diffusion for image in parallel,
//     return a combined multimodal response with both text and image content
type ModalityRoutingConfig struct {
	// Enabled activates modality routing. When false, the feature is completely skipped.
	Enabled bool `yaml:"enabled" json:"enabled"`

	// ARModel is the model name for autoregressive (text) responses.
	// Must match a key in model_config.
	ARModel string `yaml:"ar_model" json:"ar_model"`

	// AREndpoint is the base URL of the AR model's OpenAI-compatible API.
	// Required for BOTH modality (the router calls this endpoint directly for text).
	// Example: "http://localhost:8000/v1"
	AREndpoint string `yaml:"ar_endpoint" json:"ar_endpoint"`

	// DiffusionModel is the model name for diffusion (image) responses.
	DiffusionModel string `yaml:"diffusion_model" json:"diffusion_model"`

	// DiffusionEndpoint is the base URL of the diffusion model's API.
	// Example: "http://localhost:8001/v1"
	DiffusionEndpoint string `yaml:"diffusion_endpoint" json:"diffusion_endpoint"`

	// ImageGen holds image generation backend settings (size, steps, timeout, etc.).
	ImageGen ImageGenBackendConfig `yaml:"image_gen" json:"image_gen"`

	// Detection configures how prompts are classified into AR / DIFFUSION / BOTH.
	Detection ModalityDetectionConfig `yaml:"detection" json:"detection"`
}

// ImageGenBackendConfig holds the image generation backend settings used by
// modality routing when the classification is DIFFUSION.
type ImageGenBackendConfig struct {
	// Backend type: "vllm_omni", "openai"
	Backend string `yaml:"backend" json:"backend"`

	// BackendConfig is backend-specific configuration (base_url, model, etc.)
	BackendConfig interface{} `yaml:"backend_config,omitempty" json:"backend_config,omitempty"`

	// Default image dimensions (required)
	DefaultWidth  int `yaml:"default_width" json:"default_width"`
	DefaultHeight int `yaml:"default_height" json:"default_height"`

	// Timeout in seconds for image generation requests
	TimeoutSeconds int `yaml:"timeout_seconds,omitempty" json:"timeout_seconds,omitempty"`

	// ResponseText is the canned text returned alongside the generated image
	// in the Responses API format. Required when modality_routing is enabled.
	// Example: "Here is the generated image."
	ResponseText string `yaml:"response_text" json:"response_text"`

	// PromptPrefixes are prefix strings stripped from the user prompt before
	// sending it to the diffusion model (e.g. "generate an image of ", "draw ").
	// Matched case-insensitively; the first match is stripped. Optional.
	PromptPrefixes []string `yaml:"prompt_prefixes,omitempty" json:"prompt_prefixes,omitempty"`
}

// Validate validates the top-level modality routing configuration.
func (c *ModalityRoutingConfig) Validate() error {
	if c == nil || !c.Enabled {
		return nil
	}

	if c.ARModel == "" {
		return fmt.Errorf("modality_routing.ar_model is required when enabled")
	}
	if c.AREndpoint == "" {
		return fmt.Errorf("modality_routing.ar_endpoint is required when enabled (e.g. \"http://localhost:8000/v1\")")
	}
	if c.DiffusionModel == "" {
		return fmt.Errorf("modality_routing.diffusion_model is required when enabled")
	}
	if c.DiffusionEndpoint == "" {
		return fmt.Errorf("modality_routing.diffusion_endpoint is required when enabled")
	}
	if c.ImageGen.Backend == "" {
		return fmt.Errorf("modality_routing.image_gen.backend is required when enabled")
	}
	if c.ImageGen.DefaultWidth <= 0 {
		return fmt.Errorf("modality_routing.image_gen.default_width is required when enabled (e.g. 1024)")
	}
	if c.ImageGen.DefaultHeight <= 0 {
		return fmt.Errorf("modality_routing.image_gen.default_height is required when enabled (e.g. 1024)")
	}
	if c.ImageGen.ResponseText == "" {
		return fmt.Errorf("modality_routing.image_gen.response_text is required when enabled (e.g. \"Here is the generated image.\")")
	}

	// Validate detection config
	if err := c.Detection.Validate(); err != nil {
		return fmt.Errorf("modality_routing.detection: %w", err)
	}

	return nil
}

// ModalityDetectionConfig configures how modality routing detects whether a prompt
// should be routed to an AR (text) model, a Diffusion (image) model, or both.
type ModalityDetectionConfig struct {
	// Method specifies the detection strategy: "classifier", "keyword", or "hybrid" (default).
	//   - "classifier": Use mmBERT-32K ML classifier only — requires classifier.model_path
	//   - "keyword":    Use keyword pattern matching only — requires keywords list
	//   - "hybrid":     Classifier primary + keyword fallback — requires at least one of the above
	Method string `json:"method,omitempty" yaml:"method,omitempty"`

	// Classifier configuration (required when method is "classifier"; recommended for "hybrid")
	Classifier *ModalityClassifierConfig `json:"classifier,omitempty" yaml:"classifier,omitempty"`

	// Keywords for image generation detection (required when method is "keyword"; recommended for "hybrid")
	// These are matched case-insensitively against the user prompt.
	Keywords []string `json:"keywords,omitempty" yaml:"keywords,omitempty"`

	// BothKeywords are additional keywords that indicate the user wants BOTH text and image.
	// These are matched case-insensitively when the prompt also contains image keywords.
	// Examples: "explain and illustrate", "describe with a picture", "write ... with an image"
	BothKeywords []string `json:"both_keywords,omitempty" yaml:"both_keywords,omitempty"`

	// ConfidenceThreshold is the minimum classifier confidence to accept its prediction.
	// Below this threshold, the system falls back to keyword detection (hybrid mode).
	// Required when method is "classifier" or "hybrid".
	ConfidenceThreshold float32 `json:"confidence_threshold,omitempty" yaml:"confidence_threshold,omitempty"`

	// LowerThresholdRatio controls the hybrid mode disagreement behavior.
	// When classifier and keyword methods disagree, the classifier is still preferred
	// if its confidence >= confidence_threshold * lower_threshold_ratio.
	// Required when method is "hybrid". Typical value: 0.7 (i.e. 70% of confidence_threshold).
	LowerThresholdRatio float32 `json:"lower_threshold_ratio,omitempty" yaml:"lower_threshold_ratio,omitempty"`
}

// ModalityClassifierConfig configures the ML-based modality classifier
type ModalityClassifierConfig struct {
	// ModelPath is the path to the merged modality classifier model directory.
	// Required when method is "classifier" or "hybrid" with a classifier.
	ModelPath string `json:"model_path,omitempty" yaml:"model_path,omitempty"`

	// UseCPU forces CPU inference even when GPU is available.
	UseCPU bool `json:"use_cpu,omitempty" yaml:"use_cpu,omitempty"`
}

// GetMethod returns the configured modality detection method.
// Returns empty string if not set — callers should use Validate() to ensure
// Method is one of "classifier", "keyword", or "hybrid" before calling this.
func (c *ModalityDetectionConfig) GetMethod() string {
	if c == nil {
		return ""
	}
	return c.Method
}

// GetConfidenceThreshold returns the configured confidence threshold.
// This value must be explicitly set in config when method is "classifier" or "hybrid";
// Validate() enforces this requirement.
func (c *ModalityDetectionConfig) GetConfidenceThreshold() float32 {
	if c == nil {
		return 0
	}
	return c.ConfidenceThreshold
}

// GetLowerThresholdRatio returns the configured lower threshold ratio.
// This value must be explicitly set in config when method is "hybrid";
// Validate() enforces this requirement.
func (c *ModalityDetectionConfig) GetLowerThresholdRatio() float32 {
	if c == nil {
		return 0
	}
	return c.LowerThresholdRatio
}

// ImageGenPluginConfig represents configuration for image generation plugin
type ImageGenPluginConfig struct {
	// Enable image generation for this decision
	Enabled bool `json:"enabled" yaml:"enabled"`

	// Backend type: "vllm_omni", "openai", "replicate"
	Backend string `json:"backend" yaml:"backend"`

	// Backend-specific configuration
	BackendConfig interface{} `json:"backend_config,omitempty" yaml:"backend_config,omitempty"`

	// ModalityDetection configures how prompts are classified into AR/DIFFUSION/BOTH.
	// If not specified, defaults to hybrid (classifier + keyword fallback).
	ModalityDetection *ModalityDetectionConfig `json:"modality_detection,omitempty" yaml:"modality_detection,omitempty"`

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

// UnmarshalBackendConfig converts the raw BackendConfig (map from YAML) into the
// properly typed struct based on the Backend field. This must be called before
// passing the config to imagegen.CreateBackend.
func (c *ImageGenPluginConfig) UnmarshalBackendConfig() error {
	if c.BackendConfig == nil || c.Backend == "" {
		return nil
	}
	var typedConfig interface{}
	switch c.Backend {
	case "vllm_omni":
		typedConfig = &VLLMOmniImageGenConfig{}
	case "openai":
		typedConfig = &OpenAIImageGenConfig{}
	default:
		return fmt.Errorf("unknown image_gen backend: %s", c.Backend)
	}
	if err := unmarshalPluginConfig(c.BackendConfig, typedConfig); err != nil {
		return fmt.Errorf("failed to unmarshal backend config for %s: %w", c.Backend, err)
	}
	c.BackendConfig = typedConfig
	return nil
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

// Validate validates the modality detection configuration.
// It ensures that:
//   - Method (if set) is one of "classifier", "keyword", or "hybrid"
//   - For "classifier": Classifier config with a non-empty model_path is required
//   - For "keyword": At least one keyword must be configured
//   - For "hybrid": At least one of Classifier or Keywords must be configured
//   - ConfidenceThreshold (if set) is in the range (0, 1]
//   - ConfidenceThreshold is required when method is "classifier" or "hybrid"
func (c *ModalityDetectionConfig) Validate() error {
	if c == nil {
		return nil // nil config is valid (not used in top-level modality_routing path)
	}

	method := c.GetMethod()
	if method == "" {
		return fmt.Errorf("modality_detection.method is required (one of %q, %q, or %q)",
			ModalityDetectionClassifier, ModalityDetectionKeyword, ModalityDetectionHybrid)
	}

	// Validate method value
	switch method {
	case ModalityDetectionClassifier, ModalityDetectionKeyword, ModalityDetectionHybrid:
		// valid
	default:
		return fmt.Errorf("modality_detection.method must be one of %q, %q, or %q, got %q",
			ModalityDetectionClassifier, ModalityDetectionKeyword, ModalityDetectionHybrid, method)
	}

	// Method-specific validation
	switch method {
	case ModalityDetectionClassifier:
		if c.Classifier == nil || c.Classifier.ModelPath == "" {
			return fmt.Errorf("modality_detection: method %q requires classifier.model_path to be set", method)
		}

	case ModalityDetectionKeyword:
		if len(c.Keywords) == 0 {
			return fmt.Errorf("modality_detection: method %q requires at least one entry in keywords", method)
		}

	case ModalityDetectionHybrid:
		hasClassifier := c.Classifier != nil && c.Classifier.ModelPath != ""
		hasKeywords := len(c.Keywords) > 0
		if !hasClassifier && !hasKeywords {
			return fmt.Errorf("modality_detection: method %q requires at least one of classifier.model_path or keywords to be configured", method)
		}
	}

	// Validate confidence threshold
	if c.ConfidenceThreshold != 0 {
		if c.ConfidenceThreshold < 0 || c.ConfidenceThreshold > 1 {
			return fmt.Errorf("modality_detection.confidence_threshold must be between 0 and 1, got %.4f", c.ConfidenceThreshold)
		}
	}

	// confidence_threshold is required when the classifier is involved
	if (method == ModalityDetectionClassifier || method == ModalityDetectionHybrid) && c.ConfidenceThreshold == 0 {
		return fmt.Errorf("modality_detection.confidence_threshold is required when method is %q (e.g. 0.6)", method)
	}

	// lower_threshold_ratio validation
	if c.LowerThresholdRatio != 0 {
		if c.LowerThresholdRatio < 0 || c.LowerThresholdRatio > 1 {
			return fmt.Errorf("modality_detection.lower_threshold_ratio must be between 0 and 1, got %.4f", c.LowerThresholdRatio)
		}
	}
	// lower_threshold_ratio is required for hybrid (controls classifier-vs-keyword disagreement)
	if method == ModalityDetectionHybrid && c.LowerThresholdRatio == 0 {
		return fmt.Errorf("modality_detection.lower_threshold_ratio is required when method is %q (e.g. 0.7)", method)
	}

	return nil
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

	// Validate modality detection if present
	if c.ModalityDetection != nil {
		if err := c.ModalityDetection.Validate(); err != nil {
			return fmt.Errorf("image_gen: %w", err)
		}
	}

	return nil
}
