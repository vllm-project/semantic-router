package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestStreamingBug_DiffusionReturnsJSON proves that when stream:true is set
// and modality routing matches DIFFUSION, the fix ensures the response has
// content-type text/event-stream with valid SSE chunks.
func TestStreamingBug_DiffusionProducesValidSSE(t *testing.T) {
	mockDiffusion := newModalityMockBackend(t)
	defer mockDiffusion.Close()

	ctx, router := newModalityStreamingContext(t, mockDiffusion.URL, true)
	req := newModalityChatRequest("generate an image of a cat")

	resp, err := router.handleModalityFromDecision(ctx, req)
	if err != nil {
		t.Fatalf("handleModalityFromDecision failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response for DIFFUSION modality")
	}
	immResp := resp.GetImmediateResponse()
	if immResp == nil {
		t.Fatal("expected ImmediateResponse")
	}

	contentType := getHeader(immResp, "content-type")
	bodyStr := string(immResp.Body)
	t.Logf("Content-Type: %s", contentType)
	t.Logf("Body preview (first 300 bytes):\n%s", bodyStr[:min(len(bodyStr), 300)])

	assertSSEFormat(t, contentType, immResp.Body, bodyStr)
	t.Log("PASS: streaming DIFFUSION produces valid SSE response")
}

// newModalityMockBackend starts a mock vLLM-Omni backend returning
// ChatCompletion format with content array containing image_url.
func newModalityMockBackend(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{
				"message": map[string]interface{}{
					"role": "assistant",
					"content": []map[string]interface{}{{
						"type":      "image_url",
						"image_url": map[string]string{"url": "data:image/png;base64,iVBORw=="},
					}},
				},
				"finish_reason": "stop",
			}},
		})
	}))
}

// newModalityStreamingContext creates a RequestContext and RouterConfig for
// modality streaming tests. streamExpected controls ExpectStreamingResponse.
func newModalityStreamingContext(t *testing.T, backendURL string, streamExpected bool) (*RequestContext, *OpenAIRouter) {
	t.Helper()
	ctx := &RequestContext{
		RequestID:               "test-bug-001",
		ExpectStreamingResponse: streamExpected,
		TraceContext:            context.Background(),
		ModalityClassification: &ModalityClassificationResult{
			Modality: ModalityDiffusion, Confidence: 0.95, Method: "keyword",
		},
		VSRSelectedDecision: &config.Decision{
			Name:      "image_generation",
			ModelRefs: []config.ModelRef{{Model: "mock-diffusion"}},
		},
		UserContent: "generate an image of a cat",
	}
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			ModalityDetector: config.ModalityDetectorConfig{Enabled: true},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"mock-diffusion": {
					Modality: "diffusion", ImageGenBackend: "mock_backend",
					PreferredEndpoints: []string{"mock-ep"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "mock-ep", Address: "127.0.0.1", Port: 8001},
			},
			ImageGenBackends: map[string]config.ImageGenBackendEntry{
				"mock_backend": {Type: "vllm_omni", BaseURL: backendURL},
			},
		},
	}
	return ctx, &OpenAIRouter{Config: cfg}
}

func newModalityChatRequest(content string) *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage(content)},
	}
}

func getHeader(immResp *ext_proc.ImmediateResponse, key string) string {
	for _, h := range immResp.Headers.GetSetHeaders() {
		if h.Header.Key == key {
			return string(h.Header.RawValue)
		}
	}
	return ""
}

func assertSSEFormat(t *testing.T, contentType string, body []byte, bodyStr string) {
	t.Helper()
	if contentType == "" {
		t.Error("no content-type header found")
	}
	if contentType != "text/event-stream; charset=utf-8" {
		t.Errorf("stream=true expected content-type=text/event-stream; charset=utf-8, got %q", contentType)
	}
	if !bytes.HasPrefix(body, []byte("data: ")) {
		t.Errorf("stream=true body must start with 'data: ', got: %s", bodyStr[:min(len(bodyStr), 50)])
	}
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Errorf("SSE response missing termination marker 'data: [DONE]'")
	}
	if !strings.Contains(bodyStr, "chat.completion.chunk") {
		t.Errorf("SSE response missing 'chat.completion.chunk' object type")
	}
}

// TestStreamingBug_ARUnaffected verifies that AR modality is NOT affected.
func TestStreamingBug_ARUnaffected(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("what is the capital of France"),
		},
	}

	ctx := &RequestContext{
		RequestID:               "test-ar-001",
		ExpectStreamingResponse: true,
		TraceContext:            context.Background(),
		ModalityClassification: &ModalityClassificationResult{
			Modality:   ModalityAR,
			Confidence: 0.95,
			Method:     "keyword",
		},
	}

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			ModalityDetector: config.ModalityDetectorConfig{Enabled: true},
		},
		BackendModels: config.BackendModels{},
	}

	router := &OpenAIRouter{Config: cfg}

	resp, err := router.handleModalityFromDecision(ctx, req)
	if err != nil {
		t.Fatalf("handleModalityFromDecision failed: %v", err)
	}
	if resp != nil {
		t.Error("AR modality should return nil to continue normal routing, but got a response")
	}
	t.Log("AR modality correctly returns nil — streaming unaffected")
}

// TestStreamingBug_NonStreamingUnaffected verifies non-streaming requests are fine.
func TestStreamingBug_NonStreamingUnaffected(t *testing.T) {
	mockDiffusion := newModalityMockBackend(t)
	defer mockDiffusion.Close()

	ctx, router := newModalityStreamingContext(t, mockDiffusion.URL, false)
	req := newModalityChatRequest("generate an image of a cat")

	resp, err := router.handleModalityFromDecision(ctx, req)
	if err != nil {
		t.Fatalf("handleModalityFromDecision failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response for DIFFUSION")
	}

	contentType := getHeader(resp.GetImmediateResponse(), "content-type")
	t.Logf("Content-Type: %s", contentType)
	if contentType == "text/event-stream; charset=utf-8" {
		t.Errorf("non-streaming request got SSE content-type, expected application/json")
	} else {
		t.Log("non-streaming DIFFUSION correctly returns application/json")
	}
}
