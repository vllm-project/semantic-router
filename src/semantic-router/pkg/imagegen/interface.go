package imagegen

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Backend represents an image generation backend
type Backend interface {
	// Name returns the backend name (e.g., "vllm_omni", "openai", "replicate")
	Name() string

	// GenerateImage generates an image from the given request
	GenerateImage(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error)

	// HealthCheck checks if the backend is healthy
	HealthCheck(ctx context.Context) error
}

// GenerateRequest represents a request to generate an image
type GenerateRequest struct {
	// Prompt is the text description of the image to generate
	Prompt string

	// NegativePrompt is text describing what to avoid (optional)
	NegativePrompt string

	// Width of the generated image in pixels
	Width int

	// Height of the generated image in pixels
	Height int

	// NumInferenceSteps is the number of denoising steps (for diffusion models)
	NumInferenceSteps int

	// GuidanceScale controls how closely the image follows the prompt
	GuidanceScale float64

	// Seed for reproducibility (optional)
	Seed *int

	// Model to use (optional, backend-specific)
	Model string

	// Quality setting (e.g., "standard", "hd") - OpenAI specific
	Quality string

	// Style setting (e.g., "vivid", "natural") - OpenAI specific
	Style string
}

// GenerateResponse represents the response from image generation
type GenerateResponse struct {
	// ImageURL is the URL or data URI of the generated image
	ImageURL string

	// ImageBase64 is the base64-encoded image data (alternative to URL)
	ImageBase64 string

	// RevisedPrompt is the prompt that was actually used (may be modified by the backend)
	RevisedPrompt string

	// Model is the model that was used
	Model string

	// Backend is the name of the backend that generated the image
	Backend string
}

// Factory creates image generation backends based on configuration
type Factory struct {
	backends map[string]func(*config.ImageGenPluginConfig) (Backend, error)
}

// NewFactory creates a new backend factory with all registered backends
func NewFactory() *Factory {
	f := &Factory{
		backends: make(map[string]func(*config.ImageGenPluginConfig) (Backend, error)),
	}

	// Register built-in backends
	f.Register("vllm_omni", NewVLLMOmniBackend)
	f.Register("openai", NewOpenAIBackend)

	return f
}

// Register registers a new backend type
func (f *Factory) Register(name string, constructor func(*config.ImageGenPluginConfig) (Backend, error)) {
	f.backends[name] = constructor
}

// Create creates a backend instance based on the configuration
func (f *Factory) Create(cfg *config.ImageGenPluginConfig) (Backend, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	constructor, ok := f.backends[cfg.Backend]
	if !ok {
		return nil, fmt.Errorf("unknown image generation backend: %s", cfg.Backend)
	}

	return constructor(cfg)
}

// DefaultFactory is the global factory instance
var DefaultFactory = NewFactory()

// CreateBackend creates a backend using the default factory
func CreateBackend(cfg *config.ImageGenPluginConfig) (Backend, error) {
	return DefaultFactory.Create(cfg)
}
