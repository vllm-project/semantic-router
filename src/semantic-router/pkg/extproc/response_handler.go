package extproc

import (
	"encoding/json"
	"strconv"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/openai/openai-go"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// handleResponseHeaders processes the response headers
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Detect upstream HTTP status and record non-2xx as errors
	if v != nil && v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
		if statusCode := getStatusFromHeaders(v.ResponseHeaders.Headers); statusCode != 0 {
			if statusCode >= 500 {
				metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_5xx")
			} else if statusCode >= 400 {
				metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_4xx")
			}
		}
	}

	// Best-effort TTFT measurement: record on first response headers if we have a start time and model
	if ctx != nil && !ctx.TTFTRecorded && !ctx.ProcessingStartTime.IsZero() && ctx.RequestModel != "" {
		ttft := time.Since(ctx.ProcessingStartTime).Seconds()
		if ttft > 0 {
			metrics.RecordModelTTFT(ctx.RequestModel, ttft)
			ctx.TTFTSeconds = ttft
			ctx.TTFTRecorded = true
		}
	}

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

// getStatusFromHeaders extracts :status pseudo-header value as integer
func getStatusFromHeaders(headerMap *core.HeaderMap) int {
	if headerMap == nil {
		return 0
	}
	for _, hv := range headerMap.Headers {
		if hv.Key == ":status" {
			if hv.Value != "" {
				if code, err := strconv.Atoi(hv.Value); err == nil {
					return code
				}
			}
			if len(hv.RawValue) > 0 {
				if code, err := strconv.Atoi(string(hv.RawValue)); err == nil {
					return code
				}
			}
		}
	}
	return 0
}

func getModelFromCtx(ctx *RequestContext) string {
	if ctx == nil || ctx.RequestModel == "" {
		return "unknown"
	}
	return ctx.RequestModel
}

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

	// Process the response for caching
	responseBody := v.ResponseBody.Body

	// Parse tokens from the response JSON using OpenAI SDK types
	var parsed openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		observability.Errorf("Error parsing tokens from response: %v", err)
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

		// Compute and record cost if pricing is configured
		if r.Config != nil {
			promptRatePer1M, completionRatePer1M, currency, ok := r.Config.GetModelPricing(ctx.RequestModel)
			if ok {
				costAmount := (float64(promptTokens)*promptRatePer1M + float64(completionTokens)*completionRatePer1M) / 1_000_000.0
				if currency == "" {
					currency = "USD"
				}
				metrics.RecordModelCost(ctx.RequestModel, currency, costAmount)
				observability.LogEvent("llm_usage", map[string]interface{}{
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
				observability.LogEvent("llm_usage", map[string]interface{}{
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
			observability.Errorf("Error updating cache: %v", err)
			// Continue even if cache update fails
		} else {
			observability.Infof("Cache updated for request ID: %s", ctx.RequestID)
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
