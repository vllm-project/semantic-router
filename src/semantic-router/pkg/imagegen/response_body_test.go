package imagegen

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestImageGenBackendsResponseLimit(t *testing.T) {
	backends := []struct {
		name     string
		response string
		new      func(*testing.T, string, int64) Backend
	}{
		{
			name:     "openai",
			response: `{"data":[{"url":"https://example.com/image.png"}]}`,
			new: func(t *testing.T, baseURL string, maxResponseBytes int64) Backend {
				t.Helper()
				backend, err := NewOpenAIBackend(&config.ImageGenPluginConfig{
					Backend:          "openai",
					MaxResponseBytes: maxResponseBytes,
					BackendConfig: config.MustStructuredPayload(&config.OpenAIImageGenConfig{
						APIKey:  "test-key",
						BaseURL: baseURL,
					}),
				})
				if err != nil {
					t.Fatalf("NewOpenAIBackend() error = %v", err)
				}
				return backend
			},
		},
		{
			name:     "vllm_omni",
			response: `{"model":"test-model","choices":[{"message":{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,test"}}]}}]}`,
			new: func(t *testing.T, baseURL string, maxResponseBytes int64) Backend {
				t.Helper()
				backend, err := NewVLLMOmniBackend(&config.ImageGenPluginConfig{
					Backend:          "vllm_omni",
					MaxResponseBytes: maxResponseBytes,
					BackendConfig: config.MustStructuredPayload(&config.VLLMOmniImageGenConfig{
						BaseURL: baseURL,
					}),
				})
				if err != nil {
					t.Fatalf("NewVLLMOmniBackend() error = %v", err)
				}
				return backend
			},
		},
	}

	for _, backendCase := range backends {
		t.Run(backendCase.name, func(t *testing.T) {
			testImageGenBackendResponseLimit(t, backendCase.response, backendCase.new)
		})
	}
}

func testImageGenBackendResponseLimit(t *testing.T, response string, newBackend func(*testing.T, string, int64) Backend) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, response)
	}))
	defer server.Close()

	limits := []struct {
		name             string
		maxResponseBytes int64
		wantErr          bool
	}{
		{name: "default"},
		{name: "at limit", maxResponseBytes: int64(len(response))},
		{name: "one byte over", maxResponseBytes: int64(len(response)) - 1, wantErr: true},
	}
	for _, limitCase := range limits {
		t.Run(limitCase.name, func(t *testing.T) {
			backend := newBackend(t, server.URL, limitCase.maxResponseBytes)
			_, err := backend.GenerateImage(context.Background(), &GenerateRequest{Prompt: "test"})
			if limitCase.wantErr {
				if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
					t.Fatalf("GenerateImage() error = %v, want response limit error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("GenerateImage() error = %v", err)
			}
		})
	}
}

func TestReadImageGenResponseTruncatesErrorBody(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusBadGateway,
		Body:       io.NopCloser(strings.NewReader(strings.Repeat("x", int(maxImageGenErrorBodyBytes)) + "tail")),
	}

	_, err := readImageGenResponse(resp, defaultImageGenMaxResponseBytes)
	if err == nil || !strings.Contains(err.Error(), "truncated=true") {
		t.Fatalf("readImageGenResponse() error = %v, want truncated error", err)
	}
	if strings.Contains(err.Error(), "tail") {
		t.Fatalf("readImageGenResponse() included content beyond the error-body limit")
	}
}
