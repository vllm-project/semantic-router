package imagegen

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewVLLMOmniBackend(t *testing.T) {
	cfg := &config.ImageGenPluginConfig{
		Backend:        "vllm_omni",
		TimeoutSeconds: 30,
		BackendConfig: &config.VLLMOmniImageGenConfig{
			BaseURL:           "http://localhost:8001",
			Model:             "Qwen/Qwen-Image",
			NumInferenceSteps: 50,
			CFGScale:          4.0,
		},
	}

	backend, err := NewVLLMOmniBackend(cfg)
	if err != nil {
		t.Fatalf("NewVLLMOmniBackend failed: %v", err)
	}

	if backend.Name() != "vllm_omni" {
		t.Errorf("expected name vllm_omni, got %s", backend.Name())
	}
}

func TestVLLMOmniBackend_GenerateImage(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}

		// Return mock image generation response
		response := vllmOmniResponse{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "Qwen/Qwen-Image",
			Choices: []vllmOmniChoice{
				{
					Index: 0,
					Message: vllmOmniResponseMsg{
						Role: "assistant",
						Content: []map[string]interface{}{
							{
								"type": "image_url",
								"image_url": map[string]string{
									"url": "data:image/png;base64,iVBORw0KGgo=",
								},
							},
						},
					},
					FinishReason: "stop",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := &config.ImageGenPluginConfig{
		Backend:        "vllm_omni",
		TimeoutSeconds: 10,
		BackendConfig: &config.VLLMOmniImageGenConfig{
			BaseURL: server.URL,
			Model:   "Qwen/Qwen-Image",
		},
	}

	backend, err := NewVLLMOmniBackend(cfg)
	if err != nil {
		t.Fatalf("NewVLLMOmniBackend failed: %v", err)
	}

	req := &GenerateRequest{
		Prompt: "A sunset over mountains",
		Width:  512,
		Height: 512,
	}

	resp, err := backend.GenerateImage(context.Background(), req)
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}

	if resp.ImageURL != "data:image/png;base64,iVBORw0KGgo=" {
		t.Errorf("unexpected image URL: %s", resp.ImageURL)
	}
	if resp.Backend != "vllm_omni" {
		t.Errorf("expected backend vllm_omni, got %s", resp.Backend)
	}
}

func TestVLLMOmniBackend_HealthCheck(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := &config.ImageGenPluginConfig{
		Backend:        "vllm_omni",
		TimeoutSeconds: 5,
		BackendConfig: &config.VLLMOmniImageGenConfig{
			BaseURL: server.URL,
		},
	}

	backend, err := NewVLLMOmniBackend(cfg)
	if err != nil {
		t.Fatalf("NewVLLMOmniBackend failed: %v", err)
	}

	err = backend.HealthCheck(context.Background())
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
}

func TestExtractImageURL(t *testing.T) {
	tests := []struct {
		name      string
		response  *vllmOmniResponse
		wantURL   string
		wantError bool
	}{
		{
			name: "valid image response",
			response: &vllmOmniResponse{
				Choices: []vllmOmniChoice{
					{
						Message: vllmOmniResponseMsg{
							Content: []interface{}{
								map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]interface{}{
										"url": "data:image/png;base64,test123",
									},
								},
							},
						},
					},
				},
			},
			wantURL:   "data:image/png;base64,test123",
			wantError: false,
		},
		{
			name: "empty choices",
			response: &vllmOmniResponse{
				Choices: []vllmOmniChoice{},
			},
			wantError: true,
		},
		{
			name: "text only response",
			response: &vllmOmniResponse{
				Choices: []vllmOmniChoice{
					{
						Message: vllmOmniResponseMsg{
							Content: "This is just text",
						},
					},
				},
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url, err := extractImageURL(tt.response)
			if tt.wantError {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if url != tt.wantURL {
					t.Errorf("expected URL %s, got %s", tt.wantURL, url)
				}
			}
		})
	}
}
