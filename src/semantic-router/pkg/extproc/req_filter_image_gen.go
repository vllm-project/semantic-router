package extproc

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/imagegen"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Image generation detection patterns
var imageGenPatterns = regexp.MustCompile(`(?i)(generate|create|draw|make|produce|render)\s+(an?\s+)?(image|picture|photo|illustration|artwork|diagram|visualization)`)

// detectImageGenerationRequest checks if the request requires image generation
func (r *OpenAIRouter) detectImageGenerationRequest(ctx *RequestContext) bool {
	// Check 1: Request contains image_generation tool in Responses API format
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		if hasImageGenTool(ctx.OriginalRequestBody) {
			logging.Infof("[ImageGen] Detected image_generation tool in Responses API request")
			return true
		}
	}

	// Check 2: Prompt analysis for image generation keywords
	if imageGenPatterns.MatchString(ctx.UserContent) {
		logging.Infof("[ImageGen] Detected image generation intent in prompt")
		return true
	}

	return false
}

// hasImageGenTool checks if the request contains image_generation tool
func hasImageGenTool(requestBody []byte) bool {
	var req map[string]interface{}
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return false
	}

	tools, ok := req["tools"].([]interface{})
	if !ok {
		return false
	}

	for _, tool := range tools {
		toolMap, ok := tool.(map[string]interface{})
		if !ok {
			continue
		}
		if toolType, ok := toolMap["type"].(string); ok && toolType == "image_generation" {
			return true
		}
	}

	return false
}

// executeImageGenPlugin executes image generation if enabled for the decision
func (r *OpenAIRouter) executeImageGenPlugin(ctx *RequestContext, decisionName string) (*ImageGenResult, error) {
	decision := ctx.VSRSelectedDecision
	if decision == nil {
		return nil, nil
	}

	// Get image gen config
	imageGenConfig := decision.GetImageGenConfig()
	if imageGenConfig == nil || !imageGenConfig.Enabled {
		return nil, nil
	}

	// Check if this request requires image generation
	if !r.detectImageGenerationRequest(ctx) {
		return nil, nil
	}

	logging.Infof("[ImageGen] Executing image generation plugin for decision: %s", decisionName)

	// Create backend using factory
	backend, err := imagegen.CreateBackend(imageGenConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create image generation backend: %w", err)
	}

	// Build generation request
	genReq := &imagegen.GenerateRequest{
		Prompt: ctx.UserContent,
		Width:  getOrDefault(imageGenConfig.DefaultWidth, 1024),
		Height: getOrDefault(imageGenConfig.DefaultHeight, 1024),
	}

	// Execute generation
	start := time.Now()
	resp, err := backend.GenerateImage(ctx.TraceContext, genReq)
	latency := time.Since(start).Seconds()

	if err != nil {
		metrics.RecordImageGenRequest(backend.Name(), "error", latency)
		return nil, fmt.Errorf("image generation failed: %w", err)
	}

	metrics.RecordImageGenRequest(backend.Name(), "success", latency)
	logging.Infof("[ImageGen] Generated image in %.2fs via %s", latency, backend.Name())

	return &ImageGenResult{
		ImageURL:      resp.ImageURL,
		ImageBase64:   resp.ImageBase64,
		RevisedPrompt: resp.RevisedPrompt,
		Model:         resp.Model,
	}, nil
}

// ImageGenResult represents the result of image generation
type ImageGenResult struct {
	ImageURL      string `json:"image_url"`
	ImageBase64   string `json:"image_base64,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
	Model         string `json:"model,omitempty"`
}

// buildImageGenResponse builds the response with generated image
func (r *OpenAIRouter) buildImageGenResponse(result *ImageGenResult, ctx *RequestContext) ([]byte, error) {
	// Check if this is a Responses API request
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		return r.buildResponsesAPIImageResponse(result, ctx)
	}

	// Build Chat Completions format response
	return r.buildChatCompletionsImageResponse(result, ctx)
}

// buildResponsesAPIImageResponse builds Responses API format response
func (r *OpenAIRouter) buildResponsesAPIImageResponse(result *ImageGenResult, ctx *RequestContext) ([]byte, error) {
	response := map[string]interface{}{
		"id":      fmt.Sprintf("resp_%d", time.Now().UnixNano()),
		"object":  "response",
		"created": time.Now().Unix(),
		"model":   result.Model,
		"status":  "completed",
		"output": []map[string]interface{}{
			{
				"type":   "image_generation_call",
				"id":     fmt.Sprintf("ig_%d", time.Now().UnixNano()),
				"status": "completed",
				"result": result.ImageURL,
			},
			{
				"type": "message",
				"role": "assistant",
				"content": []map[string]interface{}{
					{
						"type": "output_text",
						"text": "I've generated an image based on your request.",
					},
				},
			},
		},
	}

	return json.Marshal(response)
}

// buildChatCompletionsImageResponse builds Chat Completions format response
func (r *OpenAIRouter) buildChatCompletionsImageResponse(result *ImageGenResult, ctx *RequestContext) ([]byte, error) {
	response := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   result.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role": "assistant",
					"content": []map[string]interface{}{
						{
							"type": "image_url",
							"image_url": map[string]string{
								"url": result.ImageURL,
							},
						},
					},
				},
				"finish_reason": "stop",
			},
		},
	}

	return json.Marshal(response)
}

// Helper functions
func getOrDefault(value, defaultValue int) int {
	if value == 0 {
		return defaultValue
	}
	return value
}

// ExtractImagePrompt extracts and cleans the image generation prompt
func ExtractImagePrompt(userContent string) string {
	// Remove common prefixes like "generate an image of", "create a picture of", etc.
	prompt := userContent

	// Common patterns to remove
	prefixes := []string{
		"generate an image of ",
		"create an image of ",
		"draw an image of ",
		"make an image of ",
		"generate a picture of ",
		"create a picture of ",
		"draw a picture of ",
		"generate ",
		"create ",
		"draw ",
	}

	lowerPrompt := strings.ToLower(prompt)
	for _, prefix := range prefixes {
		if strings.HasPrefix(lowerPrompt, prefix) {
			prompt = prompt[len(prefix):]
			break
		}
	}

	return strings.TrimSpace(prompt)
}
