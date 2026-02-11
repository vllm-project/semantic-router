package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Modality constants returned by the modality routing classifier
const (
	ModalityAR        = "AR"        // Text-only response via autoregressive LLM
	ModalityDiffusion = "DIFFUSION" // Image generation via diffusion model
	ModalityBoth      = "BOTH"      // Hybrid response requiring both text and image
)

// Keyword classification confidence constants.
// These represent the inherent precision of each keyword-matching scenario.
// Unlike ConfidenceThreshold (a tunable gate), these are fixed scores reflecting
// how certain the keyword method is about its classification in each case.
const (
	// keywordConfidenceToolDetected: Responses API image_generation tool explicitly present in request
	keywordConfidenceToolDetected float32 = 0.9
	// keywordConfidenceMatch: a configured keyword was found in the prompt
	keywordConfidenceMatch float32 = 0.8
	// keywordConfidenceBothMatch: image keyword + both_keyword both found
	keywordConfidenceBothMatch float32 = 0.75
	// keywordConfidenceNoMatch: no image keyword found → AR
	keywordConfidenceNoMatch float32 = 0.8
	// keywordConfidenceNoConfig: keyword method requested but keywords not configured
	keywordConfidenceNoConfig float32 = 0.5
)

// ModalityClassificationResult holds the result of modality routing classification
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", or "hybrid"
}

// classifyModality determines the response modality for the user's prompt.
// It supports three configurable methods via ModalityDetectionConfig:
//   - "classifier": ML-based (mmBERT-32K) — errors if model not loaded
//   - "keyword":    Configurable keyword matching — requires keywords in config
//   - "hybrid":     Classifier when available + keyword confirmation/fallback (default)
func (r *OpenAIRouter) classifyModality(ctx *RequestContext, detectionConfig *config.ModalityDetectionConfig) ModalityClassificationResult {
	userContent := ctx.UserContent
	if userContent == "" {
		return ModalityClassificationResult{Modality: ModalityAR, Confidence: 1.0, Method: "default"}
	}

	method := detectionConfig.GetMethod()

	var result ModalityClassificationResult

	switch method {
	case config.ModalityDetectionClassifier:
		result = r.classifyByClassifier(ctx, detectionConfig)
	case config.ModalityDetectionKeyword:
		result = r.classifyByKeyword(ctx, detectionConfig)
	case config.ModalityDetectionHybrid:
		result = r.classifyHybrid(ctx, detectionConfig)
	default:
		// This should be unreachable — Validate() rejects invalid methods at config load.
		// If we get here, it's a programming error, not something to silently work around.
		logging.Errorf("[ModalityRouter] BUG: unknown detection method %q — this should have been caught by config validation. Defaulting to AR.", method)
		result = ModalityClassificationResult{Modality: ModalityAR, Confidence: 0.0, Method: "error/unknown-method"}
	}

	// Store in context
	ctx.ModalityClassification = &result
	return result
}

// classifyByClassifier uses the mmBERT-32K ML classifier exclusively.
// If the classifier is not initialized, it logs an error and defaults to AR —
// there is no silent fallback. Fix the deployment by loading the model.
func (r *OpenAIRouter) classifyByClassifier(ctx *RequestContext, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	result, err := candle_binding.ClassifyMmBert32KModality(ctx.UserContent)
	if err == nil {
		logging.Infof("[ModalityRouter] Classifier: %s (confidence=%.3f) for prompt: %.80s",
			result.Modality, result.Confidence, ctx.UserContent)
		return ModalityClassificationResult{
			Modality:   result.Modality,
			Confidence: result.Confidence,
			Method:     "classifier",
		}
	}

	// Classifier not loaded — this is a deployment error, not something to silently work around
	logging.Errorf("[ModalityRouter] Classifier unavailable: %v — modality detection disabled, defaulting to AR. "+
		"Load the model or switch to method: \"keyword\" in modality_detection config.", err)
	return ModalityClassificationResult{Modality: ModalityAR, Confidence: 0.0, Method: "classifier/error"}
}

// classifyByKeyword uses keyword patterns from config to detect modality.
// Keywords MUST be provided in the modality_detection config — there are no hardcoded defaults.
// Supports 3-class detection: AR, DIFFUSION, and BOTH (via both_keywords).
func (r *OpenAIRouter) classifyByKeyword(ctx *RequestContext, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	// Check for Responses API image_generation tool first
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		if hasImageGenTool(ctx.OriginalRequestBody) {
			logging.Infof("[ModalityRouter] Keyword: detected image_generation tool in Responses API request")
			return ModalityClassificationResult{Modality: ModalityDiffusion, Confidence: keywordConfidenceToolDetected, Method: "keyword"}
		}
	}

	// Keywords must come from config
	if cfg == nil || len(cfg.Keywords) == 0 {
		logging.Warnf("[ModalityRouter] Keyword detection requested but no keywords configured in modality_detection.keywords — defaulting to AR")
		return ModalityClassificationResult{Modality: ModalityAR, Confidence: keywordConfidenceNoConfig, Method: "keyword/no-config"}
	}

	lowerContent := strings.ToLower(ctx.UserContent)

	// Check if any configured image keyword matches
	hasImageIntent := false
	for _, kw := range cfg.Keywords {
		if strings.Contains(lowerContent, strings.ToLower(kw)) {
			hasImageIntent = true
			break
		}
	}

	if !hasImageIntent {
		return ModalityClassificationResult{Modality: ModalityAR, Confidence: keywordConfidenceNoMatch, Method: "keyword"}
	}

	// Image intent detected — check if it's BOTH (text + image) using both_keywords from config
	if len(cfg.BothKeywords) > 0 {
		for _, kw := range cfg.BothKeywords {
			if strings.Contains(lowerContent, strings.ToLower(kw)) {
				logging.Infof("[ModalityRouter] Keyword: BOTH detected (image + both_keyword %q) for: %.80s",
					kw, ctx.UserContent)
				return ModalityClassificationResult{Modality: ModalityBoth, Confidence: keywordConfidenceBothMatch, Method: "keyword"}
			}
		}
	}

	// Pure image generation
	logging.Infof("[ModalityRouter] Keyword: DIFFUSION detected for: %.80s", ctx.UserContent)
	return ModalityClassificationResult{Modality: ModalityDiffusion, Confidence: keywordConfidenceMatch, Method: "keyword"}
}

// classifyHybrid uses the ML classifier as primary, with keyword matching as
// fallback (when classifier is unavailable) or confirmation (when classifier confidence is low).
func (r *OpenAIRouter) classifyHybrid(ctx *RequestContext, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	confThreshold := cfg.GetConfidenceThreshold()

	// Try classifier first
	classifierResult, err := candle_binding.ClassifyMmBert32KModality(ctx.UserContent)
	if err == nil && classifierResult.Confidence >= confThreshold {
		// High-confidence classifier result - trust it
		logging.Infof("[ModalityRouter] Hybrid(classifier): %s (confidence=%.3f, threshold=%.2f) for: %.80s",
			classifierResult.Modality, classifierResult.Confidence, confThreshold, ctx.UserContent)
		return ModalityClassificationResult{
			Modality:   classifierResult.Modality,
			Confidence: classifierResult.Confidence,
			Method:     "hybrid/classifier",
		}
	}

	if err == nil {
		// Classifier available but low confidence - use keyword to confirm/override
		keywordResult := r.classifyByKeyword(ctx, cfg)

		if classifierResult.Modality == keywordResult.Modality {
			// Agreement - boost confidence
			logging.Infof("[ModalityRouter] Hybrid(agree): %s (classifier=%.3f, keyword=%.3f) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Confidence, ctx.UserContent)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: (classifierResult.Confidence + keywordResult.Confidence) / 2,
				Method:     "hybrid/agree",
			}
		}

		// Disagreement - prefer classifier if above a lower threshold, else keyword
		lowerThreshold := confThreshold * cfg.GetLowerThresholdRatio()
		if classifierResult.Confidence >= lowerThreshold {
			logging.Infof("[ModalityRouter] Hybrid(classifier-preferred): %s (classifier=%.3f vs keyword=%s) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Modality, ctx.UserContent)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: classifierResult.Confidence,
				Method:     "hybrid/classifier-preferred",
			}
		}

		logging.Infof("[ModalityRouter] Hybrid(keyword-override): %s (classifier=%s@%.3f too low, keyword=%s) for: %.80s",
			keywordResult.Modality, classifierResult.Modality, classifierResult.Confidence,
			keywordResult.Modality, ctx.UserContent)
		return ModalityClassificationResult{
			Modality:   keywordResult.Modality,
			Confidence: keywordResult.Confidence,
			Method:     "hybrid/keyword-override",
		}
	}

	// Classifier unavailable - fall back to keyword detection
	logging.Debugf("[ModalityRouter] Hybrid: classifier unavailable (%v), using keyword detection", err)
	return r.classifyByKeyword(ctx, cfg)
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
						"text": result.ResponseText,
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
