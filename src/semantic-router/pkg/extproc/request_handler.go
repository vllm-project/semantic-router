package extproc

import (
	"encoding/json"
	"log"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/utils/http"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/utils/pii"
)

// parseOpenAIRequest parses the raw JSON using the OpenAI SDK types
func parseOpenAIRequest(data []byte) (*openai.ChatCompletionNewParams, error) {
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// serializeOpenAIRequest converts request back to JSON
func serializeOpenAIRequest(req *openai.ChatCompletionNewParams) ([]byte, error) {
	return json.Marshal(req)
}

// extractUserAndNonUserContent extracts content from request messages
func extractUserAndNonUserContent(req *openai.ChatCompletionNewParams) (string, []string) {
	var userContent string
	var nonUser []string

	for _, msg := range req.Messages {
		// Extract content based on message type
		var textContent string
		var role string

		if msg.OfUser != nil {
			role = "user"
			// Handle user message content
			if msg.OfUser.Content.OfString.Value != "" {
				textContent = msg.OfUser.Content.OfString.Value
			} else if len(msg.OfUser.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfUser.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfSystem != nil {
			role = "system"
			if msg.OfSystem.Content.OfString.Value != "" {
				textContent = msg.OfSystem.Content.OfString.Value
			} else if len(msg.OfSystem.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfSystem.Content.OfArrayOfContentParts {
					if part.Text != "" {
						parts = append(parts, part.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfAssistant != nil {
			role = "assistant"
			if msg.OfAssistant.Content.OfString.Value != "" {
				textContent = msg.OfAssistant.Content.OfString.Value
			} else if len(msg.OfAssistant.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfAssistant.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		}

		// Categorize by role
		if role == "user" {
			userContent = textContent
		} else if role != "" {
			nonUser = append(nonUser, textContent)
		}
	}

	return userContent, nonUser
}

// RequestContext holds the context for processing a request
type RequestContext struct {
	Headers             map[string]string
	RequestID           string
	OriginalRequestBody []byte
	RequestModel        string
	RequestQuery        string
	StartTime           time.Time
	ProcessingStartTime time.Time
}

// handleRequestHeaders processes the request headers
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Record start time for overall request processing
	ctx.StartTime = time.Now()
	log.Println("Received request headers")

	// Store headers for later use
	headers := v.RequestHeaders.Headers
	log.Printf("Processing %d request headers", len(headers.Headers))
	for _, h := range headers.Headers {
		// Prefer Value when available; fall back to RawValue
		headerValue := h.Value
		if headerValue == "" && len(h.RawValue) > 0 {
			headerValue = string(h.RawValue)
		}

		ctx.Headers[h.Key] = headerValue
		// Store request ID if present (case-insensitive)
		if strings.ToLower(h.Key) == "x-request-id" {
			ctx.RequestID = headerValue
		}
	}

	// Allow the request to continue
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					// No HeaderMutation - will be handled in body phase
				},
			},
		},
	}

	return response, nil
}

// handleRequestBody processes the request body
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	log.Println("Received request body")
	// Record start time for model routing
	ctx.ProcessingStartTime = time.Now()
	// Save the original request body
	ctx.OriginalRequestBody = v.RequestBody.Body

	// Parse the OpenAI request using SDK types
	openAIRequest, err := parseOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		log.Printf("Error parsing OpenAI request: %v", err)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}

	// Store the original model
	originalModel := string(openAIRequest.Model)
	log.Printf("Original model: %s", originalModel)

	// Record the initial request to this model
	metrics.RecordModelRequest(originalModel)

	// Get content from messages
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	// Perform security checks
	if response, shouldReturn := r.performSecurityChecks(ctx, userContent, nonUserMessages); shouldReturn {
		return response, nil
	}

	// Handle caching
	if response, shouldReturn := r.handleCaching(ctx); shouldReturn {
		return response, nil
	}

	// Handle model selection and routing
	return r.handleModelRouting(openAIRequest, originalModel, userContent, nonUserMessages, ctx)
}

// performSecurityChecks performs PII and jailbreak detection
func (r *OpenAIRouter) performSecurityChecks(ctx *RequestContext, userContent string, nonUserMessages []string) (*ext_proc.ProcessingResponse, bool) {
	// Perform PII classification on all message content
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)

	// Perform jailbreak detection on all message content
	if r.Classifier.IsJailbreakEnabled() {
		hasJailbreak, jailbreakDetections, err := r.Classifier.AnalyzeContentForJailbreak(allContent)
		if err != nil {
			log.Printf("Error performing jailbreak analysis: %v", err)
			// Continue processing despite jailbreak analysis error
		} else if hasJailbreak {
			// Find the first jailbreak detection for response
			var jailbreakType string
			var confidence float32
			for _, detection := range jailbreakDetections {
				if detection.IsJailbreak {
					jailbreakType = detection.JailbreakType
					confidence = detection.Confidence
					break
				}
			}

			log.Printf("JAILBREAK ATTEMPT BLOCKED: %s (confidence: %.3f)", jailbreakType, confidence)

			// Return immediate jailbreak violation response
			// Structured log for security block
			observability.LogEvent("security_block", map[string]interface{}{
				"reason_code":    "jailbreak_detected",
				"jailbreak_type": jailbreakType,
				"confidence":     confidence,
				"request_id":     ctx.RequestID,
			})
			jailbreakResponse := http.CreateJailbreakViolationResponse(jailbreakType, confidence)
			return jailbreakResponse, true
		} else {
			log.Printf("No jailbreak detected in request content")
		}
	}

	return nil, false
}

// handleCaching handles cache lookup and storage
func (r *OpenAIRouter) handleCaching(ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		log.Printf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	if requestQuery != "" && r.Cache.IsEnabled() {
		// Try to find a similar cached response
		cachedResponse, found, err := r.Cache.FindSimilar(requestModel, requestQuery)
		if err != nil {
			log.Printf("Error searching cache: %v", err)
		} else if found {
			// Record and log cache hit
			metrics.RecordCacheHit()
			observability.LogEvent("cache_hit", map[string]interface{}{
				"request_id": ctx.RequestID,
				"model":      requestModel,
				"query":      requestQuery,
			})
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse)
			return response, true
		}

		// Cache miss, store the request for later
		cacheID, err := r.Cache.AddPendingRequest(requestModel, requestQuery, ctx.OriginalRequestBody)
		if err != nil {
			log.Printf("Error adding pending request to cache: %v", err)
		} else {
			r.pendingRequestsLock.Lock()
			r.pendingRequests[ctx.RequestID] = []byte(cacheID)
			r.pendingRequestsLock.Unlock()
		}
	}

	return nil, false
}

// handleModelRouting handles model selection and routing logic
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel, userContent string, nonUserMessages []string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Create default response with CONTINUE status
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	// Only change the model if the original model is "auto"
	actualModel := originalModel
	var selectedEndpoint string
	if originalModel == "auto" && (len(nonUserMessages) > 0 || userContent != "") {
		log.Printf("Using Auto Model Selection")
		// Determine text to use for classification/similarity
		var classificationText string
		if len(userContent) > 0 {
			classificationText = userContent
		} else if len(nonUserMessages) > 0 {
			// Fall back to user content if no system/assistant messages
			classificationText = strings.Join(nonUserMessages, " ")
		}

		if classificationText != "" {
			// Find the most similar task description or classify, then select best model
			matchedModel := r.classifyAndSelectBestModel(classificationText)
			if matchedModel != originalModel && matchedModel != "" {
				// Get detected PII for policy checking
				allContent := pii.ExtractAllContent(userContent, nonUserMessages)
				if r.PIIChecker.IsPIIEnabled(matchedModel) {
					log.Printf("PII policy enabled for model %s", matchedModel)
					detectedPII := r.Classifier.DetectPIIInContent(allContent)

					// Check if the initially selected model passes PII policy
					allowed, deniedPII, err := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
					if err != nil {
						log.Printf("Error checking PII policy for model %s: %v", matchedModel, err)
						// Continue with original selection on error
					} else if !allowed {
						log.Printf("Initially selected model %s violates PII policy, finding alternative", matchedModel)
						// Find alternative models from the same category that pass PII policy
						categoryName := r.findCategoryForClassification(classificationText)
						if categoryName != "" {
							alternativeModels := r.Classifier.GetModelsForCategory(categoryName)
							allowedModels := r.PIIChecker.FilterModelsForPII(alternativeModels, detectedPII)
							if len(allowedModels) > 0 {
								// Select the best allowed model from this category
								matchedModel = r.Classifier.SelectBestModelFromList(allowedModels, categoryName)
								log.Printf("Selected alternative model %s that passes PII policy", matchedModel)
								// Record reason code for selecting alternative due to PII
								metrics.RecordRoutingReasonCode("pii_policy_alternative_selected", matchedModel)
							} else {
								log.Printf("No models in category %s pass PII policy, using default", categoryName)
								matchedModel = r.Config.DefaultModel
								// Check if default model passes policy
								defaultAllowed, defaultDeniedPII, _ := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
								if !defaultAllowed {
									log.Printf("Default model also violates PII policy, returning error")
									observability.LogEvent("routing_block", map[string]interface{}{
										"reason_code": "pii_policy_denied_default_model",
										"request_id":  ctx.RequestID,
										"model":       matchedModel,
										"denied_pii":  defaultDeniedPII,
									})
									piiResponse := http.CreatePIIViolationResponse(matchedModel, defaultDeniedPII)
									return piiResponse, nil
								}
							}
						} else {
							log.Printf("Could not determine category, returning PII violation for model %s", matchedModel)
							observability.LogEvent("routing_block", map[string]interface{}{
								"reason_code": "pii_policy_denied",
								"request_id":  ctx.RequestID,
								"model":       matchedModel,
								"denied_pii":  deniedPII,
							})
							piiResponse := http.CreatePIIViolationResponse(matchedModel, deniedPII)
							return piiResponse, nil
						}
					}
				}

				log.Printf("Routing to model: %s", matchedModel)

				// Check reasoning mode for this category
				useReasoning, categoryName := r.getReasoningModeAndCategory(userContent)
				log.Printf("Reasoning mode decision for this query: %v on [%s] model", useReasoning, matchedModel)
				// Record reasoning decision metric with the effort that will be applied if enabled
				effortForMetrics := r.getReasoningEffort(categoryName)
				metrics.RecordReasoningDecision(categoryName, matchedModel, useReasoning, effortForMetrics)

				// Track the model load for the selected model
				r.Classifier.IncrementModelLoad(matchedModel)

				// Track the model routing change
				metrics.RecordModelRouting(originalModel, matchedModel)

				// Update the actual model that will be used
				actualModel = matchedModel

				// Select the best endpoint for this model
				endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(matchedModel)
				if endpointFound {
					selectedEndpoint = endpointAddress
					log.Printf("Selected endpoint address: %s for model: %s", selectedEndpoint, matchedModel)
				} else {
					log.Printf("Warning: No endpoint found for model %s, using fallback", matchedModel)
				}

				// Modify the model in the request
				openAIRequest.Model = openai.ChatModel(matchedModel)

				// Serialize the modified request
				modifiedBody, err := serializeOpenAIRequest(openAIRequest)
				if err != nil {
					log.Printf("Error serializing modified request: %v", err)
					return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
				}

				modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, categoryName)
				if err != nil {
					log.Printf("Error setting reasoning mode %v to request: %v", useReasoning, err)
					return nil, status.Errorf(codes.Internal, "error setting reasoning mode: %v", err)
				}

				// Create body mutation with the modified body
				bodyMutation := &ext_proc.BodyMutation{
					Mutation: &ext_proc.BodyMutation_Body{
						Body: modifiedBody,
					},
				}

				// Create header mutation with content-length removal AND all necessary routing headers
				// (body phase HeaderMutation replaces header phase completely)
				setHeaders := []*core.HeaderValueOption{}
				if selectedEndpoint != "" {
					setHeaders = append(setHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:      "x-semantic-destination-endpoint",
							RawValue: []byte(selectedEndpoint),
						},
					})
				}
				if actualModel != "" {
					setHeaders = append(setHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:      "x-selected-model",
							Value:    actualModel,
							RawValue: []byte(actualModel),
						},
					})
				}

				headerMutation := &ext_proc.HeaderMutation{
					RemoveHeaders: []string{"content-length"},
					SetHeaders:    setHeaders,
				}

				log.Printf("ActualModel = '%s'", actualModel)

				// Set the response with body mutation and content-length removal
				response = &ext_proc.ProcessingResponse{
					Response: &ext_proc.ProcessingResponse_RequestBody{
						RequestBody: &ext_proc.BodyResponse{
							Response: &ext_proc.CommonResponse{
								Status:         ext_proc.CommonResponse_CONTINUE,
								HeaderMutation: headerMutation,
								BodyMutation:   bodyMutation,
							},
						},
					},
				}

				log.Printf("Use new model: %s", matchedModel)

				// Structured log for routing decision (auto)
				observability.LogEvent("routing_decision", map[string]interface{}{
					"reason_code":        "auto_routing",
					"request_id":         ctx.RequestID,
					"original_model":     originalModel,
					"selected_model":     matchedModel,
					"category":           categoryName,
					"reasoning_enabled":  useReasoning,
					"reasoning_effort":   effortForMetrics,
					"selected_endpoint":  selectedEndpoint,
					"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
				})
				metrics.RecordRoutingReasonCode("auto_routing", matchedModel)
			}
		}
	} else if originalModel != "auto" {
		log.Printf("Using specified model: %s", originalModel)
		// For non-auto models, check PII policy compliance
		allContent := pii.ExtractAllContent(userContent, nonUserMessages)
		detectedPII := r.Classifier.DetectPIIInContent(allContent)

		allowed, deniedPII, err := r.PIIChecker.CheckPolicy(originalModel, detectedPII)
		if err != nil {
			log.Printf("Error checking PII policy for model %s: %v", originalModel, err)
			// Continue with request on error
		} else if !allowed {
			log.Printf("Model %s violates PII policy, returning error", originalModel)
			observability.LogEvent("routing_block", map[string]interface{}{
				"reason_code": "pii_policy_denied",
				"request_id":  ctx.RequestID,
				"model":       originalModel,
				"denied_pii":  deniedPII,
			})
			piiResponse := http.CreatePIIViolationResponse(originalModel, deniedPII)
			return piiResponse, nil
		}

		// Select the best endpoint for the specified model
		endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(originalModel)
		if endpointFound {
			selectedEndpoint = endpointAddress
			log.Printf("Selected endpoint address: %s for model: %s", selectedEndpoint, originalModel)
		} else {
			log.Printf("Warning: No endpoint found for model %s, using fallback", originalModel)
		}
		setHeaders := []*core.HeaderValueOption{}
		if selectedEndpoint != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      "x-semantic-destination-endpoint",
					RawValue: []byte(selectedEndpoint),
				},
			})
		}
		// Set the response with body mutation and content-length removal
		response = &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestBody{
				RequestBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
						HeaderMutation: &ext_proc.HeaderMutation{
							SetHeaders: setHeaders,
						},
					},
				},
			},
		}
		// Structured log for routing decision (explicit model)
		observability.LogEvent("routing_decision", map[string]interface{}{
			"reason_code":        "model_specified",
			"request_id":         ctx.RequestID,
			"original_model":     originalModel,
			"selected_model":     originalModel,
			"category":           "",
			"reasoning_enabled":  false,
			"reasoning_effort":   "",
			"selected_endpoint":  selectedEndpoint,
			"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
		})
		metrics.RecordRoutingReasonCode("model_specified", originalModel)
	}

	// Save the actual model that will be used for token tracking
	ctx.RequestModel = actualModel

	// Handle tool selection based on tool_choice field
	if err := r.handleToolSelection(openAIRequest, userContent, nonUserMessages, &response, ctx); err != nil {
		log.Printf("Error in tool selection: %v", err)
		// Continue without failing the request
	}

	// Record the routing latency
	routingLatency := time.Since(ctx.ProcessingStartTime)
	metrics.RecordModelRoutingLatency(routingLatency.Seconds())

	return response, nil
}

// handleToolSelection handles automatic tool selection based on semantic similarity
func (r *OpenAIRouter) handleToolSelection(openAIRequest *openai.ChatCompletionNewParams, userContent string, nonUserMessages []string, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Check if tool_choice is set to "auto"
	if openAIRequest.ToolChoice.OfAuto.Value == "auto" {
		// Continue with tool selection logic
	} else {
		return nil // Not auto tool selection
	}

	// Get text for tools classification
	var classificationText string
	if len(userContent) > 0 {
		classificationText = userContent
	} else if len(nonUserMessages) > 0 {
		classificationText = strings.Join(nonUserMessages, " ")
	}

	if classificationText == "" {
		log.Printf("No content available for tool classification")
		return nil
	}

	if !r.ToolsDatabase.IsEnabled() {
		log.Printf("Tools database is disabled")
		return nil
	}

	// Get configuration for tool selection
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3 // Default to 3 tools
	}

	// Find similar tools based on the query
	selectedTools, err := r.ToolsDatabase.FindSimilarTools(classificationText, topK)
	if err != nil {
		if r.Config.Tools.FallbackToEmpty {
			log.Printf("Tool selection failed, falling back to no tools: %v", err)
			openAIRequest.Tools = nil
			return r.updateRequestWithTools(openAIRequest, response, ctx)
		}
		return err
	}

	if len(selectedTools) == 0 {
		if r.Config.Tools.FallbackToEmpty {
			log.Printf("No suitable tools found, falling back to no tools")
			openAIRequest.Tools = nil
		} else {
			log.Printf("No suitable tools found above threshold")
			openAIRequest.Tools = []openai.ChatCompletionToolParam{} // Empty array
		}
	} else {
		// Convert selected tools to OpenAI SDK tool format
		tools := make([]openai.ChatCompletionToolParam, len(selectedTools))
		for i, tool := range selectedTools {
			// Convert the tool to OpenAI SDK format
			toolBytes, err := json.Marshal(tool)
			if err != nil {
				return err
			}
			var sdkTool openai.ChatCompletionToolParam
			if err := json.Unmarshal(toolBytes, &sdkTool); err != nil {
				return err
			}
			tools[i] = sdkTool
		}

		openAIRequest.Tools = tools
		log.Printf("Auto-selected %d tools for query: %s", len(selectedTools), classificationText)
	}

	return r.updateRequestWithTools(openAIRequest, response, ctx)
}

// updateRequestWithTools updates the request body with the selected tools
func (r *OpenAIRouter) updateRequestWithTools(openAIRequest *openai.ChatCompletionNewParams, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Re-serialize the request with modified tools
	modifiedBody, err := serializeOpenAIRequest(openAIRequest)
	if err != nil {
		return err
	}

	// Create body mutation with the modified body
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	// Create header mutation with content-length removal AND all necessary routing headers
	// (body phase HeaderMutation replaces header phase completely)

	// Get the headers that should have been set in the main routing
	var selectedEndpoint, actualModel string

	// These should be available from the existing response
	if (*response).GetRequestBody() != nil && (*response).GetRequestBody().GetResponse() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() != nil {
		for _, header := range (*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
			switch header.Header.Key {
			case "x-semantic-destination-endpoint":
				selectedEndpoint = header.Header.Value
			case "x-selected-model":
				actualModel = header.Header.Value
			}
		}
	}

	setHeaders := []*core.HeaderValueOption{}
	if selectedEndpoint != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-semantic-destination-endpoint",
				RawValue: []byte(selectedEndpoint),
			},
		})
	}
	if actualModel != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-selected-model",
				RawValue: []byte(actualModel),
			},
		})
	}

	// Intentionally do not mutate Authorization header here

	headerMutation := &ext_proc.HeaderMutation{
		RemoveHeaders: []string{"content-length"},
		SetHeaders:    setHeaders,
	}

	// Update the response with body mutation and content-length removal
	*response = &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}

	return nil
}
