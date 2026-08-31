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
		BackendConfig: config.MustStructuredPayload(&config.VLLMOmniImageGenConfig{
			BaseURL:           "http://localhost:8001",
			Model:             "Qwen/Qwen-Image",
			NumInferenceSteps: 50,
			CFGScale:          4.0,
		}),
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
	received := make(chan vllmOmniRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}

		var request vllmOmniRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		received <- request

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
		BackendConfig: config.MustStructuredPayload(&config.VLLMOmniImageGenConfig{
			BaseURL:           server.URL,
			Model:             "Qwen/Qwen-Image",
			NumInferenceSteps: 50,
			CFGScale:          4.0,
		}),
	}

	backend, err := NewVLLMOmniBackend(cfg)
	if err != nil {
		t.Fatalf("NewVLLMOmniBackend failed: %v", err)
	}

	seed := 42
	req := &GenerateRequest{
		Prompt:            "A sunset over mountains",
		NegativePrompt:    "low quality",
		Width:             512,
		Height:            768,
		NumInferenceSteps: 28,
		GuidanceScale:     3.5,
		Seed:              &seed,
		Model:             "Qwen/Qwen-Image-override",
	}

	resp, err := backend.GenerateImage(context.Background(), req)
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}

	if resp.ImageURL != "data:image/png;base64,iVBORw0KGgo=" {
		t.Errorf("unexpected image URL: %s", resp.ImageURL)
	}
	if resp.ImageBase64 != "iVBORw0KGgo=" {
		t.Errorf("unexpected image base64: %s", resp.ImageBase64)
	}
	if resp.Backend != "vllm_omni" {
		t.Errorf("expected backend vllm_omni, got %s", resp.Backend)
	}

	assertVLLMOmniRequest(t, <-received, req, seed)
}

func TestInlineImageBase64(t *testing.T) {
	tests := []struct {
		name     string
		imageURL string
		want     string
	}{
		{name: "inline PNG", imageURL: "data:image/png;base64,aGVsbG8=", want: "aGVsbG8="},
		{name: "uppercase data URL", imageURL: "DATA:IMAGE/WEBP;BASE64,AbCdEfGh", want: "AbCdEfGh"},
		{name: "remote URL", imageURL: "https://example.com/image.png"},
		{name: "non-image data URL", imageURL: "data:text/plain;base64,aGVsbG8="},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inlineImageBase64(tt.imageURL); got != tt.want {
				t.Fatalf("inlineImageBase64(%q) = %q, want %q", tt.imageURL, got, tt.want)
			}
		})
	}
}

func assertVLLMOmniRequest(t *testing.T, request vllmOmniRequest, req *GenerateRequest, seed int) {
	t.Helper()

	if request.Model != req.Model {
		t.Errorf("model = %q, want %q", request.Model, req.Model)
	}
	if len(request.Messages) != 1 || request.Messages[0].Role != "user" || request.Messages[0].Content != req.Prompt {
		t.Fatalf("messages = %#v, want one user message containing %q", request.Messages, req.Prompt)
	}
	if request.ExtraBody == nil {
		t.Fatal("extra_body is nil")
	}
	assertVLLMOmniExtraBody(t, request.ExtraBody, req, seed)
}

func assertVLLMOmniExtraBody(t *testing.T, extraBody *vllmOmniExtraBody, req *GenerateRequest, seed int) {
	t.Helper()

	if extraBody.Width != req.Width || extraBody.Height != req.Height {
		t.Errorf("size = %dx%d, want %dx%d", extraBody.Width, extraBody.Height, req.Width, req.Height)
	}
	if extraBody.NumInferenceSteps != req.NumInferenceSteps {
		t.Errorf("num_inference_steps = %d, want %d", extraBody.NumInferenceSteps, req.NumInferenceSteps)
	}
	if extraBody.TrueCFGScale != req.GuidanceScale {
		t.Errorf("true_cfg_scale = %v, want %v", extraBody.TrueCFGScale, req.GuidanceScale)
	}
	if extraBody.Seed == nil || *extraBody.Seed != seed {
		t.Errorf("seed = %v, want %d", extraBody.Seed, seed)
	}
	if extraBody.NegativePrompt != req.NegativePrompt {
		t.Errorf("negative_prompt = %q, want %q", extraBody.NegativePrompt, req.NegativePrompt)
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
		BackendConfig: config.MustStructuredPayload(&config.VLLMOmniImageGenConfig{
			BaseURL: server.URL,
		}),
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
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if url != tt.wantURL {
				t.Errorf("expected URL %s, got %s", tt.wantURL, url)
			}
		})
	}
}
