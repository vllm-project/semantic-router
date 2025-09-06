package extproc

import (
	"encoding/json"
	"log"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/openai/openai-go"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/observability"
)

// handleResponseHeaders processes the response headers
func (r *OpenAIRouter) handleResponseHeaders(_ *ext_proc.ProcessingRequest_ResponseHeaders) (*ext_proc.ProcessingResponse, error) {

	// Allow the response to continue without modification
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	return response, nil
}

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

	// Process the response for caching
	responseBody := v.ResponseBody.Body

	// Parse tokens from the response JSON using OpenAI SDK types
	var parsed openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		log.Printf("Error parsing tokens from response: %v", err)
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
		r.Classifier.DecrementModelLoad(ctx.RequestModel)

		// Compute and record cost if pricing is configured
		if r.Config != nil {
			promptRatePer1M, completionRatePer1M, ok := r.Config.GetModelPricing(ctx.RequestModel)
			if ok {
				costUSD := (float64(promptTokens)*promptRatePer1M + float64(completionTokens)*completionRatePer1M) / 1_000_000.0
				metrics.RecordModelCostUSD(ctx.RequestModel, costUSD)
				observability.LogEvent("llm_usage", map[string]interface{}{
					"request_id":            ctx.RequestID,
					"model":                 ctx.RequestModel,
					"prompt_tokens":         promptTokens,
					"completion_tokens":     completionTokens,
					"total_tokens":          promptTokens + completionTokens,
					"completion_latency_ms": completionLatency.Milliseconds(),
					"cost_usd":              costUSD,
				})
			} else {
				observability.LogEvent("llm_usage", map[string]interface{}{
					"request_id":            ctx.RequestID,
					"model":                 ctx.RequestModel,
					"prompt_tokens":         promptTokens,
					"completion_tokens":     completionTokens,
					"total_tokens":          promptTokens + completionTokens,
					"completion_latency_ms": completionLatency.Milliseconds(),
					"cost_usd":              0.0,
					"pricing":               "not_configured",
				})
			}
		}
	}

	// Check if this request has a pending cache entry
	r.pendingRequestsLock.Lock()
	cacheID, exists := r.pendingRequests[ctx.RequestID]
	if exists {
		delete(r.pendingRequests, ctx.RequestID)
	}
	r.pendingRequestsLock.Unlock()

	// If we have a pending request, update the cache
	if exists && ctx.RequestQuery != "" && responseBody != nil {
		err := r.Cache.UpdateWithResponse(string(cacheID), responseBody)
		if err != nil {
			log.Printf("Error updating cache: %v", err)
			// Continue even if cache update fails
		} else {
			log.Printf("Cache updated for request ID: %s", ctx.RequestID)
		}
	}

	// Allow the response to continue without modification
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
