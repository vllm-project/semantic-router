package extproc

import (
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// HTTPProcessingResult contains the result of processing a request through the full vSR pipeline
// This is the HTTP-friendly equivalent of the ExtProc processing
type HTTPProcessingResult struct {
	// Classification results
	Category          string
	Confidence        float64
	DecisionName      string
	SelectedModel     string
	ReasoningEnabled  bool
	SystemPromptAdded bool

	// Security results
	IsJailbreak         bool
	JailbreakType       string
	JailbreakConfidence float32
	HasPII              bool
	PIITypes            []string
	PIIDenied           []string

	// Cache results
	CacheHit       bool
	CachedResponse []byte

	// Modified request body (with system prompt, model change, etc.)
	ModifiedBody []byte

	// Hallucination mitigation
	FactCheckNeeded     bool
	FactCheckConfidence float32
	HasToolsForVerify   bool

	// Tool selection
	ToolsSelected int

	// Blocking decision
	ShouldBlock bool
	BlockReason string
}

// ProcessHTTPRequest processes an HTTP request through the full vSR pipeline
// This is the HTTP equivalent of handleRequestBody for ExtProc
func (r *OpenAIRouter) ProcessHTTPRequest(body []byte, userContent string) (*HTTPProcessingResult, error) {
	result := &HTTPProcessingResult{}

	if r.Classifier == nil {
		logging.Warnf("HTTP adapter: Classifier not initialized")
		return result, nil
	}

	// Create a minimal request context for tracking
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		StartTime:           time.Now(),
		OriginalRequestBody: body,
	}

	// Extract model from request
	requestModel, requestQuery, extractErr := cache.ExtractQueryFromOpenAIRequest(body)
	if extractErr != nil {
		logging.Warnf("HTTP adapter: failed to extract model from request: %v", extractErr)
	} else {
		ctx.RequestModel = requestModel
		ctx.RequestQuery = requestQuery
	}

	// Parse non-user messages from body
	openAIRequest, parseErr := parseOpenAIRequest(body)
	var nonUserMessages []string
	if parseErr == nil && openAIRequest != nil {
		_, nonUserMessages = extractUserAndNonUserContent(openAIRequest)
	}

	// Step 0a: Perform fact-check classification for hallucination mitigation
	r.performFactCheckClassification(ctx, userContent)
	result.FactCheckNeeded = ctx.FactCheckNeeded
	result.FactCheckConfidence = ctx.FactCheckConfidence

	// Step 0b: Check if request has tools for fact-checking
	r.checkRequestHasTools(ctx)
	result.HasToolsForVerify = ctx.HasToolsForFactCheck

	// Step 1: Decision evaluation and model selection
	decisionName, confidence, reasoningDecision, selectedModel := r.performDecisionEvaluationAndModelSelection(
		ctx.RequestModel, userContent, nonUserMessages, ctx)

	result.DecisionName = decisionName
	result.Confidence = confidence
	result.SelectedModel = selectedModel
	result.ReasoningEnabled = reasoningDecision.UseReasoning
	result.Category = ctx.VSRSelectedCategory

	logging.Infof("HTTP adapter: decision=%s, category=%s, confidence=%.3f, model=%s",
		decisionName, result.Category, confidence, selectedModel)

	// Step 2: Jailbreak detection
	if r.Classifier.IsJailbreakEnabled() {
		jailbreakEnabled := true
		if decisionName != "" && r.Config != nil {
			jailbreakEnabled = r.Config.IsJailbreakEnabledForDecision(decisionName)
		}

		if jailbreakEnabled {
			allContent := pii.ExtractAllContent(userContent, nonUserMessages)
			threshold := r.Config.PromptGuard.Threshold
			if decisionName != "" && r.Config != nil {
				threshold = r.Config.GetJailbreakThresholdForDecision(decisionName)
			}

			hasJailbreak, jailbreakDetections, jailbreakErr := r.Classifier.AnalyzeContentForJailbreakWithThreshold(allContent, threshold)
			if jailbreakErr != nil {
				logging.Errorf("HTTP adapter: jailbreak detection error: %v", jailbreakErr)
			} else if hasJailbreak {
				result.IsJailbreak = true
				for _, detection := range jailbreakDetections {
					if detection.IsJailbreak {
						result.JailbreakType = detection.JailbreakType
						result.JailbreakConfidence = detection.Confidence
						break
					}
				}

				// Block jailbreak attempts (same behavior as ExtProc)
				result.ShouldBlock = true
				result.BlockReason = "jailbreak_detected"
				logging.Warnf("HTTP adapter: JAILBREAK BLOCKED: %s (confidence: %.3f)",
					result.JailbreakType, result.JailbreakConfidence)
				return result, nil
			}
		}
	}

	// Step 3: PII detection
	if r.PIIChecker != nil && r.PIIChecker.IsPIIEnabled(decisionName) {
		allContent := pii.ExtractAllContent(userContent, nonUserMessages)
		detectedPII := r.Classifier.DetectPIIInContent(allContent)

		if len(detectedPII) > 0 {
			result.HasPII = true
			result.PIITypes = detectedPII

			// Check PII policy
			allowed, deniedPII, piiErr := r.PIIChecker.CheckPolicy(decisionName, detectedPII)
			if piiErr != nil {
				logging.Errorf("HTTP adapter: PII policy check error: %v", piiErr)
			} else if !allowed {
				result.PIIDenied = deniedPII
				result.ShouldBlock = true
				result.BlockReason = "pii_policy_violation"
				logging.Warnf("HTTP adapter: PII BLOCKED: denied types=%v", deniedPII)
				return result, nil
			}
		}
	}

	// Step 4: Cache check
	if r.Cache != nil && r.Cache.IsEnabled() {
		cacheEnabled := r.Config.SemanticCache.Enabled
		if decisionName != "" {
			cacheEnabled = r.Config.IsCacheEnabledForDecision(decisionName)
		}

		if cacheEnabled && ctx.RequestQuery != "" {
			threshold := r.Config.GetCacheSimilarityThreshold()
			if decisionName != "" {
				threshold = r.Config.GetCacheSimilarityThresholdForDecision(decisionName)
			}

			cachedResponse, found, cacheErr := r.Cache.FindSimilarWithThreshold(ctx.RequestModel, ctx.RequestQuery, threshold)
			if cacheErr != nil {
				logging.Warnf("HTTP adapter: cache lookup error: %v", cacheErr)
			} else if found {
				result.CacheHit = true
				result.CachedResponse = cachedResponse
				logging.Infof("HTTP adapter: Cache HIT for query")
			}
		}
	}

	// Step 5: Modify request body (model change, system prompt, reasoning mode)
	modifiedBody, systemPromptAdded, modifyErr := r.modifyRequestBodyForHTTP(body, decisionName, selectedModel, reasoningDecision.UseReasoning, ctx)
	if modifyErr != nil {
		logging.Errorf("HTTP adapter: failed to modify request body: %v", modifyErr)
		// Continue with original body on error
	} else if modifiedBody != nil {
		result.ModifiedBody = modifiedBody
		result.SystemPromptAdded = systemPromptAdded
	}

	// Step 6: Tool selection (if enabled and tool_choice is "auto")
	if openAIRequest != nil && r.ToolsDatabase != nil && r.ToolsDatabase.IsEnabled() {
		toolsSelected := r.selectToolsForHTTP(openAIRequest, userContent, nonUserMessages)
		if toolsSelected > 0 {
			result.ToolsSelected = toolsSelected
			// Re-serialize the request with selected tools
			if updatedBody, serializeErr := serializeOpenAIRequestWithStream(openAIRequest, false); serializeErr == nil {
				result.ModifiedBody = updatedBody
				logging.Infof("HTTP adapter: selected %d tools for request", toolsSelected)
			}
		}
	}

	return result, nil
}

// selectToolsForHTTP performs tool selection for HTTP requests
// Returns the number of tools selected (0 if none or disabled)
func (r *OpenAIRouter) selectToolsForHTTP(openAIRequest *openai.ChatCompletionNewParams, userContent string, nonUserMessages []string) int {
	// Check if tool_choice is set to "auto"
	if openAIRequest.ToolChoice.OfAuto.Value != "auto" {
		return 0
	}

	// Get text for tools classification
	var classificationText string
	if userContent != "" {
		classificationText = userContent
	} else if len(nonUserMessages) > 0 {
		for _, msg := range nonUserMessages {
			classificationText += msg + " "
		}
	}

	if classificationText == "" {
		return 0
	}

	// Get configuration for tool selection
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3 // Default to 3 tools
	}

	// Find similar tools
	selectedTools, err := r.ToolsDatabase.FindSimilarTools(classificationText, topK)
	if err != nil {
		logging.Warnf("HTTP adapter: tool selection error: %v", err)
		return 0
	}

	if len(selectedTools) == 0 {
		// Handle fallback behavior
		if r.Config.Tools.FallbackToEmpty {
			openAIRequest.Tools = nil
		}
		return 0
	}

	// Set the selected tools on the request
	openAIRequest.Tools = selectedTools
	return len(selectedTools)
}

// modifyRequestBodyForHTTP modifies the request body with model change, system prompt, and reasoning mode
// This mirrors the logic in modifyRequestBodyForAutoRouting from processor_req_body.go
func (r *OpenAIRouter) modifyRequestBodyForHTTP(body []byte, decisionName string, selectedModel string, useReasoning bool, ctx *RequestContext) ([]byte, bool, error) {
	modifiedBody := body
	var systemPromptAdded bool
	var err error

	// Step 1: Change model if needed
	if selectedModel != "" && selectedModel != ctx.RequestModel {
		modifiedBody, err = r.changeModelInBody(modifiedBody, selectedModel)
		if err != nil {
			return nil, false, err
		}
		logging.Infof("HTTP adapter: changed model from %s to %s", ctx.RequestModel, selectedModel)
	}

	// Step 2: Set reasoning mode if decision is configured
	if decisionName != "" {
		modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, decisionName)
		if err != nil {
			logging.Warnf("HTTP adapter: failed to set reasoning mode: %v", err)
			// Continue without reasoning mode modification
		}
	}

	// Step 3: Add system prompt if configured
	if decisionName != "" {
		var promptErr error
		modifiedBody, promptErr = r.addSystemPromptIfConfigured(modifiedBody, decisionName, selectedModel, ctx)
		if promptErr != nil {
			logging.Warnf("HTTP adapter: failed to add system prompt: %v", promptErr)
			// Continue without system prompt
		} else if ctx.VSRInjectedSystemPrompt {
			systemPromptAdded = true
		}
	}

	// Only return modified body if changes were made
	if string(modifiedBody) == string(body) {
		return nil, false, nil
	}

	return modifiedBody, systemPromptAdded, nil
}

// changeModelInBody replaces the model field in the request body
func (r *OpenAIRouter) changeModelInBody(body []byte, newModel string) ([]byte, error) {
	// Parse the request
	openAIRequest, err := parseOpenAIRequest(body)
	if err != nil {
		return nil, err
	}

	// Change the model
	openAIRequest.Model = newModel

	// Serialize back (preserve stream parameter)
	return serializeOpenAIRequestWithStream(openAIRequest, false)
}

// GetConfig returns the router configuration
func (r *OpenAIRouter) GetConfig() interface{} {
	return r.Config
}
