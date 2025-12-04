package extproc

import (
	"encoding/json"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	// Process the response for caching
	responseBody := v.ResponseBody.Body

	// If this is a streaming response (e.g., SSE), record TTFT on the first body chunk
	// and skip JSON parsing/caching which are not applicable for SSE chunks.
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

		// For streaming chunks, just continue (no token parsing or cache update)
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
		err := r.Cache.UpdateWithResponse(ctx.RequestID, responseBody)
		if err != nil {
			logging.Errorf("Error updating cache: %v", err)
			// Continue even if cache update fails
		} else {
			logging.Infof("Cache updated for request ID: %s", ctx.RequestID)
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
