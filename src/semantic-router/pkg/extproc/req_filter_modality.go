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

// defaultImageResponseText is the canned text returned alongside generated images.
const defaultImageResponseText = "Here is the generated image based on your request."

// InitModalityClassifier initializes the modality routing classifier.
// This should be called during application startup if the modality model is available.
func InitModalityClassifier(modelPath string, useCPU bool) error {
	return candle_binding.InitMmBert32KModalityClassifier(modelPath, useCPU)
}

// resolveARModelEndpoint derives the HTTP endpoint URL for a named AR model
// from model_config -> preferred_endpoints -> vllm_endpoints.
// The model name must come from the matched decision's ModelRefs.
func resolveARModelEndpoint(cfg *config.RouterConfig, modelName string) (string, error) {
	if modelName == "" {
		return "", fmt.Errorf("AR model name is required (must come from decision modelRefs)")
	}

	params, ok := cfg.ModelConfig[modelName]
	if !ok {
		return "", fmt.Errorf("AR model %q not found in model_config", modelName)
	}

	if len(params.PreferredEndpoints) == 0 {
		return "", fmt.Errorf("model %q has no preferred_endpoints", modelName)
	}

	endpointName := params.PreferredEndpoints[0]
	for _, ep := range cfg.VLLMEndpoints {
		if ep.Name == endpointName {
			return fmt.Sprintf("http://%s:%d/v1", ep.Address, ep.Port), nil
		}
	}
	return "", fmt.Errorf("endpoint %q (for model %q) not found in vllm_endpoints", endpointName, modelName)
}

// resolveDiffusionBackend finds the image_gen_backend for a named diffusion model
// from model_config -> image_gen_backends.
// The model name must come from the matched decision's ModelRefs.
func resolveDiffusionBackend(cfg *config.RouterConfig, modelName string) (*config.ImageGenBackendEntry, error) {
	if modelName == "" {
		return nil, fmt.Errorf("diffusion model name is required (must come from decision modelRefs)")
	}

	params, ok := cfg.ModelConfig[modelName]
	if !ok {
		return nil, fmt.Errorf("diffusion model %q not found in model_config", modelName)
	}

	if params.ImageGenBackend == "" {
		return nil, fmt.Errorf("model %q has no image_gen_backend configured", modelName)
	}

	entry, ok := cfg.ImageGenBackends[params.ImageGenBackend]
	if !ok {
		return nil, fmt.Errorf("image_gen_backend %q (for model %q) not found in image_gen_backends", params.ImageGenBackend, modelName)
	}

	return &entry, nil
}

// resolveModalityModelsFromDecision extracts the AR and diffusion model names
// from a decision's ModelRefs by looking up each model's modality in model_config.
// Returns (arModel, diffusionModel, error). Either may be empty if the decision
// doesn't include that modality — callers should check based on the modality type.
func resolveModalityModelsFromDecision(decision *config.Decision, modelConfig map[string]config.ModelParams) (string, string, error) {
	if decision == nil {
		return "", "", fmt.Errorf("decision is nil")
	}
	if len(decision.ModelRefs) == 0 {
		return "", "", fmt.Errorf("decision %q has no modelRefs", decision.Name)
	}

	var arModel, diffusionModel string
	for _, ref := range decision.ModelRefs {
		params, ok := modelConfig[ref.Model]
		if !ok {
			continue // model not in model_config — skip
		}
		switch params.Modality {
		case "ar":
			if arModel == "" {
				arModel = ref.Model
			}
		case "diffusion":
			if diffusionModel == "" {
				diffusionModel = ref.Model
			}
		}
	}

	return arModel, diffusionModel, nil
}

// handleModalityFromDecision executes modality-based routing based on the modality signal
// that was evaluated as part of the decision engine. The signal results are stored in
// ctx.VSRMatchedModality and ctx.ModalityClassification during signal evaluation.
//
// Model selection comes from the matched decision's ModelRefs (ctx.VSRSelectedDecision),
// NOT from scanning model_config. This ensures the decision engine is the single source
// of truth for which models handle each modality.
//
// Config resolution:
//
//   - Models: decision.ModelRefs -> model_config[model].modality to classify AR vs diffusion
//
//   - Image gen backend: model_config[model].image_gen_backend -> image_gen_backends[name]
//
//   - AR endpoint: model_config[model].preferred_endpoints -> vllm_endpoints
//
//   - AR:        returns (nil, nil) — continue normal LLM routing
//
//   - DIFFUSION: generates image, returns (*ProcessingResponse, nil)
//
//   - BOTH:      calls AR for text AND diffusion for image in parallel
func (r *OpenAIRouter) handleModalityFromDecision(ctx *RequestContext, openAIRequest *openai.ChatCompletionNewParams) (*ext_proc.ProcessingResponse, error) {
	cfg := config.Get()
	if cfg == nil {
		return nil, nil
	}

	if !cfg.ModalityDetector.Enabled {
		return nil, nil // Modality detector not enabled — normal flow
	}

	// Check the modality classification from signal evaluation
	if ctx.ModalityClassification == nil || ctx.ModalityClassification.Modality == "" {
		return nil, nil // No modality signal matched — normal flow
	}

	result := *ctx.ModalityClassification

	switch result.Modality {
	case ModalityAR:
		logging.Infof("[ModalityRouter] AR (method=%s) — passthrough", result.Method)
		return nil, nil

	case ModalityDiffusion:
		logging.Infof("[ModalityRouter] DIFFUSION (method=%s) — generating image", result.Method)

		// Resolve diffusion model from the matched decision's ModelRefs
		decision := ctx.VSRSelectedDecision
		if decision == nil {
			return nil, fmt.Errorf("modality DIFFUSION matched but no decision selected")
		}
		_, diffusionModel, err := resolveModalityModelsFromDecision(decision, cfg.ModelConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve models from decision %q: %w", decision.Name, err)
		}
		if diffusionModel == "" {
			return nil, fmt.Errorf("decision %q has no diffusion model in modelRefs", decision.Name)
		}

		return r.executeDiffusion(ctx, cfg, result, diffusionModel)

	case ModalityBoth:
		logging.Infof("[ModalityRouter] BOTH (method=%s) — parallel AR + diffusion", result.Method)

		// Resolve AR and diffusion models from the matched decision's ModelRefs
		decision := ctx.VSRSelectedDecision
		if decision == nil {
			return nil, fmt.Errorf("modality BOTH matched but no decision selected")
		}
		arModel, diffusionModel, err := resolveModalityModelsFromDecision(decision, cfg.ModelConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve models from decision %q: %w", decision.Name, err)
		}
		if arModel == "" {
			return nil, fmt.Errorf("decision %q has no AR model in modelRefs", decision.Name)
		}
		if diffusionModel == "" {
			return nil, fmt.Errorf("decision %q has no diffusion model in modelRefs", decision.Name)
		}

		return r.executeBoth(ctx, cfg, openAIRequest, result, arModel, diffusionModel)

	default:
		// AR fallback for unrecognized modality
		return nil, nil
	}
}

// executeDiffusion generates an image and returns an immediate response.
// diffusionModel is the model name from the decision's ModelRefs.
func (r *OpenAIRouter) executeDiffusion(ctx *RequestContext, cfg *config.RouterConfig, result ModalityClassificationResult, diffusionModel string) (*ext_proc.ProcessingResponse, error) {
	imgResult, err := r.generateImage(ctx, cfg, diffusionModel)
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
// arModel and diffusionModel are model names from the decision's ModelRefs.
func (r *OpenAIRouter) executeBoth(ctx *RequestContext, cfg *config.RouterConfig, openAIRequest *openai.ChatCompletionNewParams, result ModalityClassificationResult, arModel, diffusionModel string) (*ext_proc.ProcessingResponse, error) {
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
		textResp, textErr = r.callARModel(ctx, cfg, openAIRequest, arModel)
	}()

	// --- Diffusion image call (parallel) ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		imgRes, imgErr = r.generateImage(ctx, cfg, diffusionModel)
	}()

	wg.Wait()

	if textErr != nil {
		logging.Errorf("[ModalityRouter] BOTH: AR call failed: %v", textErr)
	}
	if imgErr != nil {
		logging.Errorf("[ModalityRouter] BOTH: diffusion call failed: %v", imgErr)
	}
	if textErr != nil && imgErr != nil {
		return nil, fmt.Errorf("BOTH: AR failed (%w) and diffusion failed (%w)", textErr, imgErr)
	}

	responseBody, err := r.buildBothResponse(textResp, imgRes, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to build BOTH response: %w", err)
	}

	return r.buildImmediateResponseWithModality(200, responseBody, result), nil
}

// callARModel sends the user's chat completion request to the AR model endpoint
// directly and returns the parsed response.
// arModel is the model name from the decision's ModelRefs; its endpoint is resolved
// from model_config -> preferred_endpoints -> vllm_endpoints.
func (r *OpenAIRouter) callARModel(ctx *RequestContext, cfg *config.RouterConfig, openAIRequest *openai.ChatCompletionNewParams, arModel string) (map[string]interface{}, error) {
	arEndpoint, err := resolveARModelEndpoint(cfg, arModel)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve AR model endpoint: %w", err)
	}

	// Serialize the original request
	reqBody, err := json.Marshal(openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal AR request: %w", err)
	}

	// Override model to the AR model name
	var reqMap map[string]interface{}
	if unmarshalErr := json.Unmarshal(reqBody, &reqMap); unmarshalErr != nil {
		return nil, fmt.Errorf("failed to parse AR request: %w", unmarshalErr)
	}
	reqMap["model"] = arModel
	delete(reqMap, "tools")
	delete(reqMap, "tool_choice")
	reqBody, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified AR request: %w", err)
	}

	url := arEndpoint + "/chat/completions"
	logging.Infof("[ModalityRouter] BOTH: calling AR endpoint %s (model=%s)", url, arModel)

	start := time.Now()
	httpReq, err := http.NewRequestWithContext(ctx.TraceContext, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create AR request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	timeout := 120 * time.Second
	client := &http.Client{Timeout: timeout}
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
// diffusionModel is the model name from the decision's ModelRefs; its backend config
// is resolved from model_config -> image_gen_backends.
// Prompt prefixes are read from modality_detector config.
func (r *OpenAIRouter) generateImage(ctx *RequestContext, cfg *config.RouterConfig, diffusionModel string) (*ImageGenResult, error) {
	// Find the diffusion model's image_gen_backend using the decision-selected model
	backendEntry, err := resolveDiffusionBackend(cfg, diffusionModel)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve diffusion backend: %w", err)
	}

	// Convert ImageGenBackendEntry -> ImageGenPluginConfig for the imagegen factory
	pluginCfg := backendEntry.ToPluginConfig()

	backend, err := imagegen.CreateBackend(pluginCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create image generation backend: %w", err)
	}

	// Get prompt prefixes from modality_detector config
	promptPrefixes := cfg.ModalityDetector.PromptPrefixes

	genReq := &imagegen.GenerateRequest{
		Prompt: ExtractImagePrompt(ctx.UserContent, promptPrefixes),
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
		ResponseText:  defaultImageResponseText,
	}, nil
}

// buildBothResponse combines the AR text response with the diffusion image into
// a single Chat Completions response containing multi-part content (text + image_url).
func (r *OpenAIRouter) buildBothResponse(textResp map[string]interface{}, imgResult *ImageGenResult, ctx *RequestContext) ([]byte, error) {
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

	modalityValue := result.Modality
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
