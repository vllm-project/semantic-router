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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
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
					Key:      headers.VSRSelectedCategory,
					RawValue: []byte(ctx.VSRSelectedCategory),
				},
			})
		}

		// Add x-vsr-selected-reasoning header
		if ctx.VSRReasoningMode != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRSelectedReasoning,
					RawValue: []byte(ctx.VSRReasoningMode),
				},
			})
		}

		// Add x-vsr-selected-model header
		if ctx.VSRSelectedModel != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRSelectedModel,
					RawValue: []byte(ctx.VSRSelectedModel),
				},
			})
		}

		// Add x-vsr-injected-system-prompt header
		injectedValue := "false"
		if ctx.VSRInjectedSystemPrompt {
			injectedValue = "true"
		}
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRInjectedSystemPrompt,
				RawValue: []byte(injectedValue),
			},
		})

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

		// If Responses adapter is active for this request, translate SSE chunks
		if r.Config != nil && r.Config.EnableResponsesAdapter {
			if p, ok := ctx.Headers[":path"]; ok && strings.HasPrefix(p, "/v1/responses") {
				body := v.ResponseBody.Body
				// Envoy provides raw chunk bytes, typically like: "data: {json}\n\n" or "data: [DONE]\n\n"
				b := string(body)
				if strings.Contains(b, "[DONE]") {
					// Emit a final response.completed if not already concluded
					response := &ext_proc.ProcessingResponse{
						Response: &ext_proc.ProcessingResponse_ResponseBody{
							ResponseBody: &ext_proc.BodyResponse{
								Response: &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE},
							},
						},
					}
					metrics.ResponsesAdapterSSEEvents.WithLabelValues("response.completed").Inc()
					return response, nil
				}

				// Extract JSON after "data: " prefix if present
				idx := strings.Index(b, "data:")
				var payload []byte
				if idx >= 0 {
					payload = []byte(strings.TrimSpace(b[idx+5:]))
				} else {
					payload = v.ResponseBody.Body
				}

				if len(payload) > 0 && payload[0] == '{' {
					if !ctx.ResponsesStreamInit {
						// Emit an initial created event on first chunk
						ctx.ResponsesStreamInit = true
						// We don't inject a new chunk here; clients will see deltas below
					}
					events, ok := translateSSEChunkToResponses(payload)
					if ok && len(events) > 0 {
						// Rebuild body as multiple SSE events in Responses format
						var sb strings.Builder
						for _, ev := range events {
							sb.WriteString("data: ")
							sb.Write(ev)
							sb.WriteString("\n\n")
							// Inspect the event type for metrics
							var et map[string]interface{}
							if err := json.Unmarshal(ev, &et); err == nil {
								if t, _ := et["type"].(string); t != "" {
									metrics.ResponsesAdapterSSEEvents.WithLabelValues(t).Inc()
								}
							}
						}
						v.ResponseBody.Body = []byte(sb.String())
					}
				}
			}
		}

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

	// If this was a /v1/responses request (adapter path), remap non-stream body to Responses JSON
	if r.Config != nil && r.Config.EnableResponsesAdapter {
		if p, ok := ctx.Headers[":path"]; ok && strings.HasPrefix(p, "/v1/responses") {
			mapped, err := mapChatCompletionToResponses(responseBody)
			if err == nil {
				// Replace upstream JSON with Responses JSON
				v.ResponseBody.Body = mapped
				// Ensure content-type remains application/json
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
		}
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
