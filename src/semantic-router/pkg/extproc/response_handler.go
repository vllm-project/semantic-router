package extproc

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/openai/openai-go"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// handleResponseHeaders processes the response headers
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	var statusCode int
	var isSuccessful bool

	// Detect upstream HTTP status and record non-2xx as errors
	if v != nil && v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
		// Determine if the response is streaming based on Content-Type
		ctx.IsStreamingResponse = isStreamingContentType(v.ResponseHeaders.Headers)

		statusCode = getStatusFromHeaders(v.ResponseHeaders.Headers)
		isSuccessful = statusCode >= 200 && statusCode < 300

		if statusCode != 0 {
			if statusCode >= 500 {
				metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_5xx")
			} else if statusCode >= 400 {
				metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_4xx")
			}
		}
	}

	// Best-effort TTFT measurement:
	// - For non-streaming responses, record on first response headers (approx TTFB ~= TTFT)
	// - For streaming responses (SSE), defer TTFT until the first response body chunk arrives
	if ctx != nil && !ctx.IsStreamingResponse && !ctx.TTFTRecorded && !ctx.ProcessingStartTime.IsZero() && ctx.RequestModel != "" {
		ttft := time.Since(ctx.ProcessingStartTime).Seconds()
		if ttft > 0 {
			metrics.RecordModelTTFT(ctx.RequestModel, ttft)
			ctx.TTFTSeconds = ttft
			ctx.TTFTRecorded = true
		}
	}

	// Prepare response headers with VSR decision tracking headers if applicable
	var headerMutation *ext_proc.HeaderMutation

	// Add VSR decision headers if request was successful and didn't hit cache
	if isSuccessful && !ctx.VSRCacheHit && ctx != nil {
		var setHeaders []*core.HeaderValueOption

		// Add x-vsr-selected-category header
		if ctx.VSRSelectedCategory != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      "x-vsr-selected-category",
					RawValue: []byte(ctx.VSRSelectedCategory),
				},
			})
		}

		// Add x-vsr-selected-reasoning header
		if ctx.VSRReasoningMode != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      "x-vsr-selected-reasoning",
					RawValue: []byte(ctx.VSRReasoningMode),
				},
			})
		}

		// Add x-vsr-selected-model header
		if ctx.VSRSelectedModel != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      "x-vsr-selected-model",
					RawValue: []byte(ctx.VSRSelectedModel),
				},
			})
		}

		// Create header mutation if we have headers to add
		if len(setHeaders) > 0 {
			headerMutation = &ext_proc.HeaderMutation{
				SetHeaders: setHeaders,
			}
		}
	}

	// Allow the response to continue with VSR headers if applicable
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
				},
			},
		},
	}

	// If this is a streaming (SSE) response, instruct Envoy to stream the response body to ExtProc
	// so we can capture TTFT on the first body chunk. Requires allow_mode_override: true in Envoy config.
	if ctx != nil && ctx.IsStreamingResponse {
		response.ModeOverride = &http_ext.ProcessingMode{
			ResponseBodyMode: http_ext.ProcessingMode_STREAMED,
		}
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

// isStreamingContentType checks if the response content-type indicates streaming (SSE)
func isStreamingContentType(headerMap *core.HeaderMap) bool {
	if headerMap == nil {
		return false
	}
	for _, hv := range headerMap.Headers {
		if strings.ToLower(hv.Key) == "content-type" {
			val := hv.Value
			if val == "" && len(hv.RawValue) > 0 {
				val = string(hv.RawValue)
			}
			if strings.Contains(strings.ToLower(val), "text/event-stream") {
				return true
			}
		}
	}
	return false
}

// handleResponseBody processes the response body
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

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
				observability.Infof("Recorded TTFT on first streamed body chunk: %.3fs", ttft)
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

	// Update the cache
	if ctx.RequestID != "" && responseBody != nil {
		err := r.Cache.UpdateWithResponse(ctx.RequestID, responseBody)
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
