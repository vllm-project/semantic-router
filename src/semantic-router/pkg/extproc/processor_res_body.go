package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	// If this is a looper internal request, skip all processing and just continue
	// The response will be handled by the looper client directly
	if ctx.LooperRequest {
		logging.Debugf("[Looper] Skipping response body processing for internal request")
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ResponseBody{
				ResponseBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}, nil
	}

	// Process the response for caching
	responseBody := v.ResponseBody.Body

	// Transform Anthropic API response to OpenAI format if this is an Anthropic-routed request
	anthropicTransformed := false
	if ctx.APIFormat == config.APIFormatAnthropic {
		transformedBody, err := anthropic.ToOpenAIResponseBody(responseBody, ctx.RequestModel)
		if err != nil {
			logging.Errorf("Failed to transform Anthropic response to OpenAI format: %v", err)
			// Return error response to client
			return r.createErrorResponse(502, fmt.Sprintf("Response transformation error: %v", err)), nil
		}
		logging.Infof("Transformed Anthropic response to OpenAI format, original size: %d, transformed size: %d",
			len(responseBody), len(transformedBody))
		responseBody = transformedBody
		anthropicTransformed = true
	}

	// If this is a streaming response (e.g., SSE), record TTFT on the first body chunk
	// and accumulate chunks for caching when stream completes.
	if ctx.IsStreamingResponse {
		if ctx != nil && !ctx.TTFTRecorded && !ctx.ProcessingStartTime.IsZero() && ctx.RequestModel != "" {
			ttft := time.Since(ctx.ProcessingStartTime).Seconds()
			if ttft > 0 {
				metrics.RecordModelTTFT(ctx.RequestModel, ttft)
				ctx.TTFTSeconds = ttft
				ctx.TTFTRecorded = true
				logging.Infof("Recorded TTFT on first streamed body chunk: %.3fs", ttft)
			}
		}

		// Accumulate streaming chunks for caching
		chunk := string(responseBody)
		if ctx.StreamingChunks == nil {
			ctx.StreamingChunks = make([]string, 0)
			ctx.StreamingMetadata = make(map[string]interface{})
		}
		ctx.StreamingChunks = append(ctx.StreamingChunks, chunk)

		// Parse chunk to extract content and metadata
		r.parseStreamingChunk(chunk, ctx)

		// Check for [DONE] marker - stream is complete
		if strings.Contains(chunk, "data: [DONE]") {
			ctx.StreamingComplete = true
			logging.Infof("Streaming response completed, attempting to cache")

			// Record completion latency for streaming responses
			if ctx.RequestModel != "" && !ctx.StartTime.IsZero() {
				completionLatency := time.Since(ctx.StartTime).Seconds()
				metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency)
				logging.Infof("Recorded completion latency for streaming response: model=%s, latency=%.3fs",
					ctx.RequestModel, completionLatency)
			}

			// Reconstruct and cache the complete response
			if err := r.cacheStreamingResponse(ctx); err != nil {
				logging.Errorf("Failed to cache streaming response: %v", err)
				// Continue even if caching fails
			}

			// For replay logging, attach the reconstructed assistant content if enabled
			replayPayload := []byte(ctx.StreamingContent)
			r.attachRouterReplayResponse(ctx, replayPayload, true)
		}

		// For streaming chunks, just continue (chunks are forwarded immediately)
		response := &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ResponseBody{
				ResponseBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
		return response, nil
	}

	// Parse tokens from the response JSON using OpenAI SDK types
	var parsed openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		logging.Errorf("Error parsing tokens from response: %v", err)
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
	}
	promptTokens := int(parsed.Usage.PromptTokens)
	completionTokens := int(parsed.Usage.CompletionTokens)

	// Record tokens used with the model that was used
	if ctx.RequestModel != "" {
		metrics.RecordModelTokensDetailed(
			ctx.RequestModel,
			float64(promptTokens),
			float64(completionTokens),
		)
		metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())

		// Record TPOT (time per output token) if completion tokens are available
		if completionTokens > 0 {
			timePerToken := completionLatency.Seconds() / float64(completionTokens)
			metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
			// Update latency classifier cache for real-time routing decisions
			// Note: ctx.RequestModel should match the model name used in decision ModelRefs
			// (either ModelRef.Model or ModelRef.LoRAName, depending on selection)
			// UpdateTPOT will trim whitespace to ensure canonical matching
			logging.Debugf("Updating TPOT cache for model: %q, TPOT: %.4f", ctx.RequestModel, timePerToken)
			classification.UpdateTPOT(ctx.RequestModel, timePerToken)
		}

		// Record windowed model metrics for load balancing
		metrics.RecordModelWindowedRequest(
			ctx.RequestModel,
			completionLatency.Seconds(),
			int64(promptTokens),
			int64(completionTokens),
			false, // isError
			false, // isTimeout
		)

		// Compute and record cost if pricing is configured
		if r.Config != nil {
			promptRatePer1M, completionRatePer1M, currency, ok := r.Config.GetModelPricing(ctx.RequestModel)
			if ok {
				costAmount := (float64(promptTokens)*promptRatePer1M + float64(completionTokens)*completionRatePer1M) / 1_000_000.0
				if currency == "" {
					currency = "USD"
				}
				metrics.RecordModelCost(ctx.RequestModel, currency, costAmount)
				logging.LogEvent("llm_usage", map[string]interface{}{
					"request_id":            ctx.RequestID,
					"model":                 ctx.RequestModel,
					"prompt_tokens":         promptTokens,
					"completion_tokens":     completionTokens,
					"total_tokens":          promptTokens + completionTokens,
					"completion_latency_ms": completionLatency.Milliseconds(),
					"cost":                  costAmount,
					"currency":              currency,
				})
			} else {
				logging.LogEvent("llm_usage", map[string]interface{}{
					"request_id":            ctx.RequestID,
					"model":                 ctx.RequestModel,
					"prompt_tokens":         promptTokens,
					"completion_tokens":     completionTokens,
					"total_tokens":          promptTokens + completionTokens,
					"completion_latency_ms": completionLatency.Milliseconds(),
					"cost":                  0.0,
					"currency":              "unknown",
					"pricing":               "not_configured",
				})
			}
		}
	}

	// Update the cache
	if ctx.RequestID != "" && responseBody != nil {
		// Get decision-specific TTL; handle nil router config gracefully
		ttlSeconds := -1 // use cache default when Config is not available
		if r != nil && r.Config != nil {
			ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
		}
		err := r.Cache.UpdateWithResponse(ctx.RequestID, responseBody, ttlSeconds)
		if err != nil {
			logging.Errorf("Error updating cache: %v", err)
			// Continue even if cache update fails
		} else {
			logging.Infof("Cache updated for request ID: %s", ctx.RequestID)
		}
	}

	// Translate response for Response API requests
	finalBody := responseBody
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && r.ResponseAPIFilter != nil {
		translatedBody, err := r.ResponseAPIFilter.TranslateResponse(ctx.TraceContext, ctx.ResponseAPICtx, responseBody)
		if err != nil {
			logging.Errorf("Response API translation error: %v", err)
			// Continue with original response on error
		} else {
			finalBody = translatedBody
			logging.Infof("Response API: Translated response to Response API format")
		}
	}

	// Build response with possible body modification
	var bodyMutation *ext_proc.BodyMutation
	var headerMutation *ext_proc.HeaderMutation

	// Set body mutation if response was transformed (Anthropic or Response API)
	if anthropicTransformed || (ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest) {
		bodyMutation = &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: finalBody,
			},
		}
		// Remove content-length so Envoy recalculates it for the modified body
		headerMutation = &ext_proc.HeaderMutation{
			RemoveHeaders: []string{"content-length"},
		}
	}

	// Perform hallucination detection if enabled and conditions are met
	if hallucinationResponse := r.performHallucinationDetection(ctx, responseBody); hallucinationResponse != nil {
		// Hallucination detected and action is "block" - return error response
		return hallucinationResponse, nil
	}

	// Check unverified factual response if hallucination plugin is enabled
	if ctx.VSRSelectedDecision != nil {
		hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
		if hallucinationConfig != nil && hallucinationConfig.Enabled {
			r.checkUnverifiedFactualResponse(ctx)
		}
	}

	// Track if body needs to be modified
	modifiedBody := responseBody
	needsBodyMutation := false

	// Apply hallucination warning (may modify body or headers)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseBody{
			ResponseBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}

	// Apply hallucination warning based on configured action
	if ctx.HallucinationDetected {
		modifiedBody, response = r.applyHallucinationWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}

	// Apply unverified factual warning based on configured action
	if ctx.UnverifiedFactualResponse {
		modifiedBody, response = r.applyUnverifiedFactualWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}

	// If body was modified, update the response with body mutation
	if needsBodyMutation {
		bodyResponse := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
		bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: modifiedBody,
			},
		}
	}

	// Capture replay response payload if enabled
	r.attachRouterReplayResponse(ctx, finalBody, true)

	return response, nil
}

// parseStreamingChunk parses an SSE chunk to extract content and metadata
func (r *OpenAIRouter) parseStreamingChunk(chunk string, ctx *RequestContext) {
	// Parse SSE format: "data: {...}\n\n"
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)

			// Skip [DONE] marker
			if data == "[DONE]" {
				continue
			}

			// Parse JSON chunk
			var chunkData map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunkData); err != nil {
				// Skip malformed JSON chunks
				continue
			}

			// Extract metadata from first chunk (id, model, created)
			if ctx.StreamingMetadata["id"] == nil {
				if id, ok := chunkData["id"].(string); ok {
					ctx.StreamingMetadata["id"] = id
				}
				if model, ok := chunkData["model"].(string); ok {
					ctx.StreamingMetadata["model"] = model
				}
				if created, ok := chunkData["created"].(float64); ok {
					ctx.StreamingMetadata["created"] = int64(created)
				}
			}

			// Extract content from delta
			if choices, ok := chunkData["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if content, ok := delta["content"].(string); ok && content != "" {
							ctx.StreamingContent += content
						}
					}
					// Extract finish_reason if present
					if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
						ctx.StreamingMetadata["finish_reason"] = finishReason
					}
				}
			}

			// Extract usage information if present (usually in final chunk)
			if usage, ok := chunkData["usage"].(map[string]interface{}); ok {
				ctx.StreamingMetadata["usage"] = usage
			}
		}
	}
}

// cacheStreamingResponse reconstructs a ChatCompletion from accumulated chunks and caches it
func (r *OpenAIRouter) cacheStreamingResponse(ctx *RequestContext) error {
	// Safety check 1: Only cache if completed normally
	if !ctx.StreamingComplete {
		logging.Warnf("Stream not completed (no [DONE] marker), skipping cache")
		return nil
	}

	// Safety check 2: Don't cache if aborted
	if ctx.StreamingAborted {
		logging.Warnf("Stream was aborted, skipping cache")
		return nil
	}

	// Safety check 3: Validate we have content
	if ctx.StreamingContent == "" {
		logging.Warnf("Streaming response has no content, skipping cache")
		return nil
	}

	// Safety check 4: Validate we have metadata
	if ctx.StreamingMetadata["id"] == nil || ctx.StreamingMetadata["model"] == nil {
		logging.Warnf("Streaming response missing required metadata, skipping cache")
		return nil
	}

	// Get finish_reason (default to "stop" if not present)
	finishReason := "stop"
	if fr, ok := ctx.StreamingMetadata["finish_reason"].(string); ok && fr != "" {
		finishReason = fr
	}

	// Extract usage information if available (usually in final chunk)
	usage := openai.CompletionUsage{
		PromptTokens:     0, // Default to zero if not available
		CompletionTokens: 0, // Default to zero if not available
		TotalTokens:      0, // Default to zero if not available
	}
	if usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{}); ok {
		if promptTokens, ok := usageMap["prompt_tokens"].(float64); ok {
			usage.PromptTokens = int64(promptTokens)
		}
		if completionTokens, ok := usageMap["completion_tokens"].(float64); ok {
			usage.CompletionTokens = int64(completionTokens)
		}
		if totalTokens, ok := usageMap["total_tokens"].(float64); ok {
			usage.TotalTokens = int64(totalTokens)
		}
	}

	// Record token metrics for streaming responses
	if ctx.RequestModel != "" && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		metrics.RecordModelTokensDetailed(
			ctx.RequestModel,
			float64(usage.PromptTokens),
			float64(usage.CompletionTokens),
		)
		logging.Infof("Recorded token metrics for streaming response: model=%s, prompt=%d, completion=%d",
			ctx.RequestModel, usage.PromptTokens, usage.CompletionTokens)

		// Record TPOT for streaming responses if completion tokens are available
		if usage.CompletionTokens > 0 && !ctx.StartTime.IsZero() {
			completionLatency := time.Since(ctx.StartTime).Seconds()
			timePerToken := completionLatency / float64(usage.CompletionTokens)
			metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
			logging.Infof("Recorded TPOT for streaming response: model=%s, TPOT=%.4f", ctx.RequestModel, timePerToken)
			classification.UpdateTPOT(ctx.RequestModel, timePerToken)
		}
	}

	// Reconstruct ChatCompletion JSON
	reconstructed := openai.ChatCompletion{
		ID:      ctx.StreamingMetadata["id"].(string),
		Object:  "chat.completion",
		Created: ctx.StreamingMetadata["created"].(int64),
		Model:   ctx.StreamingMetadata["model"].(string),
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: ctx.StreamingContent,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage, // Use extracted usage or zero values
	}

	// Marshal to JSON
	reconstructedJSON, err := json.Marshal(reconstructed)
	if err != nil {
		logging.Errorf("Failed to marshal reconstructed response: %v", err)
		return err
	}

	// Safety check 5: Validate reconstructed structure
	if len(reconstructed.Choices) == 0 || reconstructed.Choices[0].Message.Content == "" {
		logging.Warnf("Reconstructed response has no valid choices or content, skipping cache")
		return nil
	}

	// Cache the reconstructed response
	// Use AddEntry if we have all required information (works even if AddPendingRequest failed)
	// Otherwise fall back to UpdateWithResponse (requires pending request)
	if ctx.RequestID != "" && ctx.RequestQuery != "" && ctx.RequestModel != "" {
		// We have all info needed for AddEntry - use it directly
		// This works even if AddPendingRequest failed due to embedding errors
		requestBody := ctx.OriginalRequestBody
		if requestBody == nil {
			// If we don't have the original request body, create a minimal one
			// This is a fallback - ideally we'd have the original body
			requestBody = []byte("{}")
		}
		// Get decision-specific TTL
		ttlSeconds := r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
		err = r.Cache.AddEntry(ctx.RequestID, ctx.RequestModel, ctx.RequestQuery, requestBody, reconstructedJSON, ttlSeconds)
		if err != nil {
			logging.Errorf("Error caching streaming response with AddEntry: %v", err)
			// Fall back to UpdateWithResponse in case AddEntry fails
			err = r.Cache.UpdateWithResponse(ctx.RequestID, reconstructedJSON, ttlSeconds)
			if err != nil {
				logging.Errorf("Error caching streaming response with UpdateWithResponse: %v", err)
				return err
			}
			logging.Infof("Successfully cached streaming response (via UpdateWithResponse) for request ID: %s", ctx.RequestID)
		} else {
			logging.Infof("Successfully cached streaming response (via AddEntry) for request ID: %s", ctx.RequestID)
		}
	} else if ctx.RequestID != "" {
		// Fall back to UpdateWithResponse if we don't have query/model
		// Get decision-specific TTL
		ttlSeconds := r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
		err = r.Cache.UpdateWithResponse(ctx.RequestID, reconstructedJSON, ttlSeconds)
		if err != nil {
			logging.Errorf("Error caching streaming response: %v", err)
			return err
		}
		logging.Infof("Successfully cached streaming response for request ID: %s", ctx.RequestID)
	} else {
		logging.Warnf("No request ID available, cannot cache streaming response")
	}

	return nil
}
