package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/imagegen"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// InitModalityClassifier initializes the modality routing classifier.
// This should be called during application startup if the modality model is available.
func InitModalityClassifier(modelPath string, useCPU bool) error {
	return candle_binding.InitMmBert32KModalityClassifier(modelPath, useCPU)
}

// handleModalityRouting is the single entry point for modality-based routing.
// It reads the top-level modality_routing config and:
//   - AR:        returns (nil, nil) — continue normal LLM routing
//   - DIFFUSION: generates image, returns (*ProcessingResponse, nil)
//   - BOTH:      calls AR for text AND diffusion for image in parallel,
//     returns a combined multimodal response (*ProcessingResponse, nil)
func (r *OpenAIRouter) handleModalityRouting(ctx *RequestContext, openAIRequest *openai.ChatCompletionNewParams) (*ext_proc.ProcessingResponse, error) {
	cfg := config.Get()
	mr := cfg.ModalityRouting
	if mr == nil || !mr.Enabled {
		return nil, nil // Feature disabled — normal flow
	}

	// Classify
	result := r.classifyModality(ctx, &mr.Detection)

	// Store classification on the request context so response-header phase can
	// add the x-vsr-selected-modality header for AR (which goes through the
	// normal upstream flow). DIFFUSION and BOTH are short-circuited below and
	// the header is added directly to the immediate response.
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality:   string(result.Modality),
		Method:     result.Method,
		Confidence: result.Confidence,
	}

	switch result.Modality {
	case ModalityAR:
		logging.Infof("[ModalityRouter] AR (confidence=%.3f, method=%s) — passthrough to %s",
			result.Confidence, result.Method, mr.ARModel)
		return nil, nil

	case ModalityDiffusion:
		logging.Infof("[ModalityRouter] DIFFUSION (confidence=%.3f, method=%s) — generating image via %s",
			result.Confidence, result.Method, mr.DiffusionModel)
		return r.executeDiffusion(ctx, mr, result)

	case ModalityBoth:
		logging.Infof("[ModalityRouter] BOTH (confidence=%.3f, method=%s) — parallel: AR(%s) + diffusion(%s)",
			result.Confidence, result.Method, mr.ARModel, mr.DiffusionModel)
		return r.executeBoth(ctx, mr, openAIRequest, result)

	default:
		logging.Errorf("[ModalityRouter] BUG: unexpected modality %q", result.Modality)
		return nil, nil
	}
}

// executeDiffusion generates an image and returns an immediate response.
func (r *OpenAIRouter) executeDiffusion(ctx *RequestContext, mr *config.ModalityRoutingConfig, result ModalityClassificationResult) (*ext_proc.ProcessingResponse, error) {
	imgResult, err := r.generateImage(ctx, mr)
	if err != nil {
		return nil, err
	}

	responseBody, err := r.buildImageGenResponse(imgResult, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to build image response: %w", err)
	}

	return r.buildImmediateResponseWithModality(200, responseBody, result), nil
}

// executeBoth calls the AR model for text and the diffusion model for image
// generation in parallel, then combines them into a single multimodal response.
func (r *OpenAIRouter) executeBoth(ctx *RequestContext, mr *config.ModalityRoutingConfig, openAIRequest *openai.ChatCompletionNewParams, result ModalityClassificationResult) (*ext_proc.ProcessingResponse, error) {
	var (
		wg       sync.WaitGroup
		textResp map[string]interface{}
		textErr  error
		imgRes   *ImageGenResult
		imgErr   error
	)

	// --- AR text call (parallel) ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		textResp, textErr = r.callARModel(ctx, mr, openAIRequest)
	}()

	// --- Diffusion image call (parallel) ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		imgRes, imgErr = r.generateImage(ctx, mr)
	}()

	wg.Wait()

	// If text call failed, still try to return image-only
	if textErr != nil {
		logging.Errorf("[ModalityRouter] BOTH: AR call failed: %v", textErr)
	}
	// If image call failed, still try to return text-only
	if imgErr != nil {
		logging.Errorf("[ModalityRouter] BOTH: diffusion call failed: %v", imgErr)
	}
	// If both failed, return error
	if textErr != nil && imgErr != nil {
		return nil, fmt.Errorf("BOTH: AR failed (%v) and diffusion failed (%v)", textErr, imgErr)
	}

	responseBody, err := r.buildBothResponse(textResp, imgRes, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to build BOTH response: %w", err)
	}

	return r.buildImmediateResponseWithModality(200, responseBody, result), nil
}

// callARModel sends the user's chat completion request to the AR model endpoint
// directly and returns the parsed response.
func (r *OpenAIRouter) callARModel(ctx *RequestContext, mr *config.ModalityRoutingConfig, openAIRequest *openai.ChatCompletionNewParams) (map[string]interface{}, error) {
	// Serialize the original request (without tool injection)
	reqBody, err := json.Marshal(openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal AR request: %w", err)
	}

	// Override model to the AR model name
	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqBody, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to parse AR request: %w", err)
	}
	reqMap["model"] = mr.ARModel
	// Remove tools/tool_choice — we want pure text from the AR model
	delete(reqMap, "tools")
	delete(reqMap, "tool_choice")
	reqBody, _ = json.Marshal(reqMap)

	url := mr.AREndpoint + "/chat/completions"
	logging.Infof("[ModalityRouter] BOTH: calling AR endpoint %s (model=%s)", url, mr.ARModel)

	start := time.Now()
	httpReq, err := http.NewRequestWithContext(ctx.TraceContext, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create AR request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("AR request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	latency := time.Since(start).Seconds()

	if err != nil {
		return nil, fmt.Errorf("failed to read AR response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("AR endpoint returned status %d: %s", resp.StatusCode, string(body[:min(len(body), 200)]))
	}

	logging.Infof("[ModalityRouter] BOTH: AR responded in %.2fs (%d bytes)", latency, len(body))

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse AR response: %w", err)
	}

	return result, nil
}

// generateImage creates the diffusion backend, sends the prompt, and returns the result.
// Shared by executeDiffusion and executeBoth.
func (r *OpenAIRouter) generateImage(ctx *RequestContext, mr *config.ModalityRoutingConfig) (*ImageGenResult, error) {
	pluginCfg := &config.ImageGenPluginConfig{
		Enabled:       true,
		Backend:       mr.ImageGen.Backend,
		BackendConfig: mr.ImageGen.BackendConfig,
		DefaultWidth:  mr.ImageGen.DefaultWidth,
		DefaultHeight: mr.ImageGen.DefaultHeight,
	}

	if err := pluginCfg.UnmarshalBackendConfig(); err != nil {
		return nil, fmt.Errorf("failed to unmarshal backend config: %w", err)
	}

	backend, err := imagegen.CreateBackend(pluginCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create image generation backend: %w", err)
	}

	genReq := &imagegen.GenerateRequest{
		Prompt: ExtractImagePrompt(ctx.UserContent, mr.ImageGen.PromptPrefixes),
		Width:  pluginCfg.DefaultWidth,
		Height: pluginCfg.DefaultHeight,
	}

	start := time.Now()
	genResp, err := backend.GenerateImage(ctx.TraceContext, genReq)
	latency := time.Since(start).Seconds()

	if err != nil {
		metrics.RecordImageGenRequest(backend.Name(), "error", latency)
		return nil, fmt.Errorf("image generation failed: %w", err)
	}

	metrics.RecordImageGenRequest(backend.Name(), "success", latency)
	logging.Infof("[ModalityRouter] Generated image in %.2fs via %s", latency, backend.Name())

	return &ImageGenResult{
		ImageURL:      genResp.ImageURL,
		ImageBase64:   genResp.ImageBase64,
		RevisedPrompt: genResp.RevisedPrompt,
		Model:         genResp.Model,
		ResponseText:  mr.ImageGen.ResponseText,
	}, nil
}

// buildBothResponse combines the AR text response with the diffusion image into
// a single Chat Completions response containing multi-part content (text + image_url).
func (r *OpenAIRouter) buildBothResponse(textResp map[string]interface{}, imgResult *ImageGenResult, ctx *RequestContext) ([]byte, error) {
	// Extract text content from the AR response
	textContent := ""
	arModel := ""
	if textResp != nil {
		if choices, ok := textResp["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if msg, ok := choice["message"].(map[string]interface{}); ok {
					if c, ok := msg["content"].(string); ok {
						textContent = c
					}
				}
			}
		}
		if m, ok := textResp["model"].(string); ok {
			arModel = m
		}
	}

	// Build multi-part content: text first, then image
	var contentParts []map[string]interface{}

	if textContent != "" {
		contentParts = append(contentParts, map[string]interface{}{
			"type": "text",
			"text": textContent,
		})
	}

	if imgResult != nil && imgResult.ImageURL != "" {
		contentParts = append(contentParts, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]string{
				"url": imgResult.ImageURL,
			},
		})
	}

	// Fall back: if we somehow have neither, return an error message
	if len(contentParts) == 0 {
		contentParts = append(contentParts, map[string]interface{}{
			"type": "text",
			"text": "Failed to generate both text and image responses.",
		})
	}

	model := arModel
	if model == "" && imgResult != nil {
		model = imgResult.Model
	}

	response := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": contentParts,
				},
				"finish_reason": "stop",
			},
		},
	}

	return json.Marshal(response)
}

// buildImmediateResponseWithModality creates a JSON immediate response and adds
// the x-vsr-selected-modality header.
func (r *OpenAIRouter) buildImmediateResponseWithModality(statusCode int, body []byte, result ModalityClassificationResult) *ext_proc.ProcessingResponse {
	procResp := r.createJSONResponseWithBody(statusCode, body)

	modalityValue := string(result.Modality)
	if result.Method != "" {
		modalityValue += ";" + result.Method
	}
	imm := procResp.Response.(*ext_proc.ProcessingResponse_ImmediateResponse).ImmediateResponse
	imm.Headers.SetHeaders = append(imm.Headers.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.VSRSelectedModality,
			RawValue: []byte(modalityValue),
		},
	})

	return procResp
}
