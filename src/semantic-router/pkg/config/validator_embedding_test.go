package config

import (
	"strings"
	"testing"
)

func TestValidateEmbeddingContracts_NilConfigIsTolerant(t *testing.T) {
	if err := validateEmbeddingContracts(nil); err != nil {
		t.Errorf("nil config should not produce an error, got: %v", err)
	}
}

func TestValidateEmbeddingContracts_AcceptsRemoteEmbeddingProvider(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				EmbeddingConfig: HNSWConfig{
					Backend:         EmbeddingBackendOpenAICompatible,
					ModelType:       EmbeddingModelTypeRemote,
					TargetDimension: 1536,
				},
				Endpoint: EmbeddingEndpointConfig{
					BaseURL:    "https://example.test/v1",
					Model:      "text-embedding-3-small",
					APIKeyEnv:  "OPENAI_API_KEY",
					Dimensions: 1536,
				},
			},
		},
	}
	if err := validateEmbeddingContracts(cfg); err != nil {
		t.Fatalf("validateEmbeddingContracts() error = %v", err)
	}
}

func TestValidateEmbeddingContracts_RejectsInvalidRemoteEmbeddingProvider(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				EmbeddingConfig: HNSWConfig{
					Backend:         EmbeddingBackendOpenAICompatible,
					ModelType:       "multimodal",
					TargetDimension: 768,
				},
				Endpoint: EmbeddingEndpointConfig{
					BaseURL:          "example.test/v1",
					Dimensions:       1536,
					TimeoutSeconds:   -1,
					MaxRetries:       -1,
					MaxResponseBytes: -1,
				},
			},
		},
	}

	err := validateEmbeddingContracts(cfg)
	if err == nil {
		t.Fatal("expected validateEmbeddingContracts to reject invalid remote provider, got nil")
	}
	msg := err.Error()
	for _, want := range []string{
		"endpoint.base_url",
		"endpoint.model",
		"endpoint.timeout_seconds",
		"endpoint.max_retries",
		"endpoint.max_response_bytes",
		"endpoint.dimensions (1536)",
		"embedding_config.model_type",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("error should contain %q, got: %s", want, msg)
		}
	}
}

func TestValidateMmBertModelPath_RejectsClassicBERT(t *testing.T) {
	err := validateMmBertModelPath("models/mom-embedding-light")
	if err == nil {
		t.Fatal("expected error when mmbert_model_path points to a classic BERT model, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"mom-embedding-light", "classic BERT", "Vela-1.0-Encoder-307M-Embedding", "all-MiniLM-L12-v2"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error should contain %q, got: %s", want, msg)
		}
	}
}

func TestValidateMmBertModelPath_AcceptsModernBERT(t *testing.T) {
	if err := validateMmBertModelPath("models/mmbert-embed-32k-2d-matryoshka"); err != nil {
		t.Errorf("mmbert-embed-32k-2d-matryoshka should be accepted, got: %v", err)
	}
}

func TestValidateMmBertModelPath_AcceptsEmpty(t *testing.T) {
	if err := validateMmBertModelPath(""); err != nil {
		t.Errorf("empty path should be accepted, got: %v", err)
	}
}

func TestValidateMmBertModelPath_AcceptsUnknownPath(t *testing.T) {
	if err := validateMmBertModelPath("models/some-custom-model"); err != nil {
		t.Errorf("unknown model path should be accepted (not in registry), got: %v", err)
	}
}

func TestValidateEmbeddingContracts_RejectsClassicBERTInMmBertSlot(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				MmBertModelPath: "models/mom-embedding-light",
				EmbeddingConfig: HNSWConfig{ModelType: "mmbert"},
			},
		},
	}
	err := validateEmbeddingContracts(cfg)
	if err == nil {
		t.Fatal("expected validateEmbeddingContracts to reject classic BERT in mmbert slot, got nil")
	}
	if !strings.Contains(err.Error(), "classic BERT") {
		t.Errorf("error should mention 'classic BERT', got: %s", err.Error())
	}
}
