package imagegen

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFactory_Create(t *testing.T) {
	factory := NewFactory()

	tests := []struct {
		name        string
		config      *config.ImageGenPluginConfig
		wantBackend string
		wantError   bool
	}{
		{
			name: "vllm_omni backend",
			config: &config.ImageGenPluginConfig{
				Backend:        "vllm_omni",
				TimeoutSeconds: 30,
				BackendConfig: &config.VLLMOmniImageGenConfig{
					BaseURL: "http://localhost:8001",
					Model:   "test-model",
				},
			},
			wantBackend: "vllm_omni",
			wantError:   false,
		},
		{
			name: "openai backend",
			config: &config.ImageGenPluginConfig{
				Backend:        "openai",
				TimeoutSeconds: 30,
				BackendConfig: &config.OpenAIImageGenConfig{
					APIKey: "test-api-key",
					Model:  "gpt-image-1",
				},
			},
			wantBackend: "openai",
			wantError:   false,
		},
		{
			name: "unknown backend",
			config: &config.ImageGenPluginConfig{
				Backend: "unknown",
			},
			wantError: true,
		},
		{
			name:      "nil config",
			config:    nil,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend, err := factory.Create(tt.config)
			if tt.wantError {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if backend.Name() != tt.wantBackend {
				t.Errorf("expected backend %s, got %s", tt.wantBackend, backend.Name())
			}
		})
	}
}

func TestDefaultFactory(t *testing.T) {
	// Test that default factory has registered backends
	cfg := &config.ImageGenPluginConfig{
		Backend:        "vllm_omni",
		TimeoutSeconds: 10,
		BackendConfig: &config.VLLMOmniImageGenConfig{
			BaseURL: "http://localhost:8001",
		},
	}

	backend, err := CreateBackend(cfg)
	if err != nil {
		t.Fatalf("CreateBackend failed: %v", err)
	}

	if backend.Name() != "vllm_omni" {
		t.Errorf("expected vllm_omni, got %s", backend.Name())
	}
}

func TestFactory_Register(t *testing.T) {
	factory := NewFactory()

	// Register a custom backend
	factory.Register("custom", func(cfg *config.ImageGenPluginConfig) (Backend, error) {
		return &mockBackend{name: "custom"}, nil
	})

	cfg := &config.ImageGenPluginConfig{
		Backend: "custom",
	}

	backend, err := factory.Create(cfg)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	if backend.Name() != "custom" {
		t.Errorf("expected custom, got %s", backend.Name())
	}
}

// mockBackend for testing
type mockBackend struct {
	name string
}

func (m *mockBackend) Name() string {
	return m.name
}

func (m *mockBackend) GenerateImage(_ context.Context, _ *GenerateRequest) (*GenerateResponse, error) {
	return &GenerateResponse{Backend: m.name}, nil
}

func (m *mockBackend) HealthCheck(_ context.Context) error {
	return nil
}
