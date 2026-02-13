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

// ModalityModels holds the resolved model names for each modality role from a decision.
type ModalityModels struct {
	ARModel        string // Text-only autoregressive model
	DiffusionModel string // Image generation via diffusion backend
	OmniModel      string // Omni model that can handle both text and image
}

// resolveModalityModelsFromDecision extracts the AR, diffusion, and omni model names
// from a decision's ModelRefs by looking up each model's modality in model_config.
// Returns a ModalityModels struct and an error. Fields may be empty if the decision
// doesn't include that modality — callers should check based on the modality type.
func resolveModalityModelsFromDecision(decision *config.Decision, modelConfig map[string]config.ModelParams) (string, string, error) {
	if decision == nil {
		return "", "", fmt.Errorf("decision is nil")
	}
	if len(decision.ModelRefs) == 0 {
		return "", "", fmt.Errorf("decision %q has no modelRefs", decision.Name)
	}
	models := resolveAllModalityModels(decision, modelConfig)
	return models.ARModel, models.DiffusionModel, nil
}

// resolveAllModalityModels extracts all modality model names including omni.
func resolveAllModalityModels(decision *config.Decision, modelConfig map[string]config.ModelParams) ModalityModels {
	if decision == nil || len(decision.ModelRefs) == 0 {
		return ModalityModels{}
	}

	var models ModalityModels
	for _, ref := range decision.ModelRefs {
		params, ok := modelConfig[ref.Model]
		if !ok {
			continue // model not in model_config — skip
		}
		switch params.Modality {
		case "ar":
			if models.ARModel == "" {
				models.ARModel = ref.Model
			}
		case "diffusion":
			if models.DiffusionModel == "" {
				models.DiffusionModel = ref.Model
			}
		case "omni":
			if models.OmniModel == "" {
				models.OmniModel = ref.Model
			}
		}
	}

	return models
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

	// Resolve all modality models (AR, diffusion, omni) from the decision
	decision := ctx.VSRSelectedDecision
	var models ModalityModels
	if decision != nil {
		models = resolveAllModalityModels(decision, cfg.ModelConfig)
	}

	switch result.Modality {
	case ModalityAR:
		logging.Infof("[ModalityRouter] AR (method=%s) — passthrough", result.Method)
		return nil, nil

	case ModalityDiffusion:
		logging.Infof("[ModalityRouter] DIFFUSION (method=%s) — generating image", result.Method)

		if decision == nil {
			return nil, fmt.Errorf("modality DIFFUSION matched but no decision selected")
		}

		// Prefer omni model if available (can handle image generation natively)
		if models.OmniModel != "" {
			logging.Infof("[ModalityRouter] DIFFUSION: using omni model %s for image generation", models.OmniModel)
			return r.executeOmni(ctx, cfg, openAIRequest, result, models.OmniModel)
		}

		if models.DiffusionModel == "" {
			return nil, fmt.Errorf("decision %q has no diffusion or omni model in modelRefs", decision.Name)
		}

		return r.executeDiffusion(ctx, cfg, result, models.DiffusionModel)

	case ModalityBoth:
		logging.Infof("[ModalityRouter] BOTH (method=%s)", result.Method)

		if decision == nil {
			return nil, fmt.Errorf("modality BOTH matched but no decision selected")
		}

		// Prefer omni model: single call handles both text and image
		if models.OmniModel != "" {
			logging.Infof("[ModalityRouter] BOTH: using omni model %s (single call for text+image)", models.OmniModel)
			return r.executeOmni(ctx, cfg, openAIRequest, result, models.OmniModel)
		}

		// Fallback: parallel AR + diffusion calls
		logging.Infof("[ModalityRouter] BOTH: parallel AR + diffusion")
		if models.ARModel == "" {
			return nil, fmt.Errorf("decision %q has no AR or omni model in modelRefs", decision.Name)
		}
		if models.DiffusionModel == "" {
			return nil, fmt.Errorf("decision %q has no diffusion or omni model in modelRefs", decision.Name)
		}

		return r.executeBoth(ctx, cfg, openAIRequest, result, models.ARModel, models.DiffusionModel)

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

// executeOmni sends a single request to an omni model endpoint that can handle both
// text and image generation natively (e.g. vllm-omni serving Qwen2.5-Omni).
// The omni model processes the request as a whole and returns a combined response
// containing text and/or images, without needing separate AR + diffusion calls.
func (r *OpenAIRouter) executeOmni(ctx *RequestContext, cfg *config.RouterConfig, openAIRequest *openai.ChatCompletionNewParams, result ModalityClassificationResult, omniModel string) (*ext_proc.ProcessingResponse, error) {
	omniEndpoint, err := resolveARModelEndpoint(cfg, omniModel)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve omni model endpoint: %w", err)
	}

	// Serialize the original request
	reqBody, err := json.Marshal(openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal omni request: %w", err)
	}

	// Override model to the omni model name and inject image generation extra_body params
	var reqMap map[string]interface{}
	if unmarshalErr := json.Unmarshal(reqBody, &reqMap); unmarshalErr != nil {
		return nil, fmt.Errorf("failed to parse omni request: %w", unmarshalErr)
	}
	reqMap["model"] = omniModel

	// If image_generation tool params are present (from Responses API), apply them
	// as extra_body parameters for vllm-omni diffusion control.
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.ImageGenToolParams != nil {
		params := ctx.ResponseAPICtx.ImageGenToolParams
		extraBody := map[string]interface{}{}

		// Parse size string (e.g. "1024x1536") into width/height
		if params.Size != "" && params.Size != "auto" {
			w, h := parseSizeString(params.Size)
			if w > 0 && h > 0 {
				extraBody["width"] = w
				extraBody["height"] = h
			}
		}

		if len(extraBody) > 0 {
			reqMap["extra_body"] = extraBody
		}
	}

	reqBody, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified omni request: %w", err)
	}

	url := omniEndpoint + "/chat/completions"
	logging.Infof("[ModalityRouter] OMNI: calling endpoint %s (model=%s)", url, omniModel)

	start := time.Now()
	httpReq, err := http.NewRequestWithContext(ctx.TraceContext, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create omni request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	timeout := 120 * time.Second
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("omni request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	latency := time.Since(start).Seconds()

	if err != nil {
		return nil, fmt.Errorf("failed to read omni response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("omni endpoint returned status %d: %s", resp.StatusCode, string(body[:min(len(body), 200)]))
	}

	logging.Infof("[ModalityRouter] OMNI: responded in %.2fs (%d bytes)", latency, len(body))

	// Parse the omni response to extract text and image parts
	var omniResp map[string]interface{}
	if err := json.Unmarshal(body, &omniResp); err != nil {
		return nil, fmt.Errorf("failed to parse omni response: %w", err)
	}

	// Build response: if this is a Responses API request, format as Responses API;
	// otherwise return the raw Chat Completions response.
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		responseBody, err := r.buildOmniResponsesAPIResponse(omniResp, ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to build omni Responses API response: %w", err)
		}
		return r.buildImmediateResponseWithModality(200, responseBody, result), nil
	}

	// For Chat Completions API, return the raw omni model response
	return r.buildImmediateResponseWithModality(200, body, result), nil
}

// buildOmniResponsesAPIResponse builds a Responses API format response from an omni model's
// Chat Completions response. It extracts text and image content parts and formats them
// as output items, including image_generation_call items for any generated images.
func (r *OpenAIRouter) buildOmniResponsesAPIResponse(omniResp map[string]interface{}, ctx *RequestContext) ([]byte, error) {
	var outputItems []map[string]interface{}
	model := ""
	if m, ok := omniResp["model"].(string); ok {
		model = m
	}

	// Extract content from choices
	if choices, ok := omniResp["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				textParts, imageParts := extractOmniContentParts(msg["content"])

				// Add image_generation_call items for each image
				for _, imgURL := range imageParts {
					outputItems = append(outputItems, map[string]interface{}{
						"type":   "image_generation_call",
						"id":     fmt.Sprintf("ig_%d", time.Now().UnixNano()),
						"status": "completed",
						"result": imgURL,
					})
				}

				// Add message item with text content
				if len(textParts) > 0 {
					var contentParts []map[string]interface{}
					for _, text := range textParts {
						contentParts = append(contentParts, map[string]interface{}{
							"type":        "output_text",
							"text":        text,
							"annotations": []interface{}{},
						})
					}
					outputItems = append(outputItems, map[string]interface{}{
						"type":    "message",
						"id":      fmt.Sprintf("msg_%d", time.Now().UnixNano()),
						"role":    "assistant",
						"status":  "completed",
						"content": contentParts,
					})
				}
			}
		}
	}

	// If no output items were extracted, add a fallback message
	if len(outputItems) == 0 {
		outputItems = append(outputItems, map[string]interface{}{
			"type": "message",
			"id":   fmt.Sprintf("msg_%d", time.Now().UnixNano()),
			"role": "assistant",
			"content": []map[string]interface{}{
				{
					"type":        "output_text",
					"text":        defaultImageResponseText,
					"annotations": []interface{}{},
				},
			},
		})
	}

	response := map[string]interface{}{
		"id":      fmt.Sprintf("resp_%d", time.Now().UnixNano()),
		"object":  "response",
		"created": time.Now().Unix(),
		"model":   model,
		"status":  "completed",
		"output":  outputItems,
	}

	return json.Marshal(response)
}

// extractOmniContentParts extracts text strings and image URLs from a message content.
// The content can be a string, or an array of content parts (text + image_url).
func extractOmniContentParts(content interface{}) (texts []string, imageURLs []string) {
	switch v := content.(type) {
	case string:
		if v != "" {
			texts = append(texts, v)
		}
	case []interface{}:
		for _, part := range v {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch partMap["type"] {
			case "text":
				if text, ok := partMap["text"].(string); ok && text != "" {
					texts = append(texts, text)
				}
			case "image_url":
				if imageURL, ok := partMap["image_url"].(map[string]interface{}); ok {
					if url, ok := imageURL["url"].(string); ok && url != "" {
						imageURLs = append(imageURLs, url)
					}
				}
			}
		}
	}
	return
}

// parseSizeString parses an OpenAI image size string like "1024x1536" into width and height.
func parseSizeString(size string) (int, int) {
	switch size {
	case "1024x1024":
		return 1024, 1024
	case "1024x1536":
		return 1024, 1536
	case "1536x1024":
		return 1536, 1024
	default:
		return 0, 0
	}
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
// If the Responses API request included image_generation tool params, those are used
// to override default width/height/quality.
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

	// Apply image_generation tool params if present (from Responses API)
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.ImageGenToolParams != nil {
		params := ctx.ResponseAPICtx.ImageGenToolParams

		// Override size from tool params
		if params.Size != "" && params.Size != "auto" {
			w, h := parseSizeString(params.Size)
			if w > 0 && h > 0 {
				genReq.Width = w
				genReq.Height = h
			}
		}

		// Pass quality through for OpenAI backend
		if params.Quality != "" && params.Quality != "auto" {
			genReq.Quality = params.Quality
		}

		// Override model if specified in tool params
		if params.Model != "" {
			genReq.Model = params.Model
		}
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
