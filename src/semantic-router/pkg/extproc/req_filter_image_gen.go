package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Modality constants returned by the modality routing classifier
const (
	ModalityAR        = "AR"        // Text-only response via autoregressive LLM
	ModalityDiffusion = "DIFFUSION" // Image generation via diffusion model
	ModalityBoth      = "BOTH"      // Hybrid response requiring both text and image
)

// ModalityClassificationResult holds the result of modality routing classification.
// Classification logic lives in the signal evaluator (classifier.go);
// this struct is shared across the extproc package for context passing and response building.
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", "hybrid", or "signal"
}

// ImageGenResult represents the result of image generation
type ImageGenResult struct {
	ImageURL      string `json:"image_url"`
	ImageBase64   string `json:"image_base64,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
	Model         string `json:"model,omitempty"`
	ResponseText  string `json:"response_text,omitempty"` // Canned text for Responses API (from config)
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

// buildResponsesAPIImageResponse builds Responses API format response with
// image_generation_call output items matching the OpenAI spec.
// Per the spec, ImageGenerationCall has: id, type, status, result (base64 image data).
// If only a URL is available (no base64), the result field contains the URL as a fallback.
func (r *OpenAIRouter) buildResponsesAPIImageResponse(result *ImageGenResult, ctx *RequestContext) ([]byte, error) {
	// Use base64 data if available, otherwise fall back to URL
	imageResult := result.ImageBase64
	if imageResult == "" {
		imageResult = result.ImageURL
	}

	now := time.Now()
	outputItems := []map[string]interface{}{
		{
			"type":   "image_generation_call",
			"id":     fmt.Sprintf("ig_%d", now.UnixNano()),
			"status": "completed",
			"result": imageResult,
		},
	}

	// Only add the text message if there's meaningful response text
	if result.ResponseText != "" {
		outputItems = append(outputItems, map[string]interface{}{
			"type":   "message",
			"id":     fmt.Sprintf("msg_%d", now.UnixNano()),
			"role":   "assistant",
			"status": "completed",
			"content": []map[string]interface{}{
				{
					"type":        "output_text",
					"text":        result.ResponseText,
					"annotations": []interface{}{},
				},
			},
		})
	}

	response := map[string]interface{}{
		"id":         fmt.Sprintf("resp_%d", now.UnixNano()),
		"object":     "response",
		"created_at": now.Unix(),
		"model":      result.Model,
		"status":     "completed",
		"output":     outputItems,
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

// ExtractImagePrompt strips configured prefixes from the user prompt before
// sending it to the diffusion model. Prefixes come from config; if none are
// configured, the prompt is returned as-is (no hardcoded defaults).
func ExtractImagePrompt(userContent string, prefixes []string) string {
	prompt := userContent
	if len(prefixes) == 0 {
		return strings.TrimSpace(prompt)
	}

	lowerPrompt := strings.ToLower(prompt)
	for _, prefix := range prefixes {
		lowerPrefix := strings.ToLower(prefix)
		if strings.HasPrefix(lowerPrompt, lowerPrefix) {
			prompt = prompt[len(lowerPrefix):]
			break
		}
	}

	return strings.TrimSpace(prompt)
}

// setModalityFromSignals sets the ModalityClassification on the request context
// based on the modality signal evaluation results. This is called after
// EvaluateAllSignals to populate ctx.ModalityClassification for response headers.
func (r *OpenAIRouter) setModalityFromSignals(ctx *RequestContext, matchedModalityRules []string) {
	if len(matchedModalityRules) == 0 {
		return
	}
	// Use the first matched modality signal
	modality := matchedModalityRules[0]
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality:   modality,
		Confidence: 0, // Confidence is set by the signal evaluator; here we just need the modality name
		Method:     "signal",
	}
}
