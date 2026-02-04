package extproc

import (
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// handleResponseHeaders processes the response headers
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// If this is a looper internal request, update router replay status and continue
	if ctx.LooperRequest {
		// Extract status code from response headers
		statusCode := 200 // Default
		if v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
			statusCode = getStatusFromHeaders(v.ResponseHeaders.Headers)
		}

		// Update router replay with status code
		r.updateRouterReplayStatus(ctx, statusCode, false)

		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ResponseHeaders{
				ResponseHeaders: &ext_proc.HeadersResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}, nil
	}

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

	// End upstream request span (started in createRoutingResponse/createSpecifiedModelResponse)
	if ctx != nil && ctx.UpstreamSpan != nil {
		// Add response status to span
		tracing.SetSpanAttributes(ctx.UpstreamSpan,
			attribute.Int("http.status_code", statusCode))

		// Mark span as error if response was not successful
		if !isSuccessful && statusCode != 0 {
			ctx.UpstreamSpan.SetStatus(codes.Error, "upstream request failed")
		}

		ctx.UpstreamSpan.End()
		ctx.UpstreamSpan = nil
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
			// Update TTFT cache for latency signal evaluation
			classification.UpdateTTFT(ctx.RequestModel, ttft)
		}
	}

	// Update router replay metadata with status code and streaming flag
	r.updateRouterReplayStatus(ctx, statusCode, ctx != nil && ctx.IsStreamingResponse)

	// Prepare response headers with VSR decision tracking headers if applicable
	var headerMutation *ext_proc.HeaderMutation

	// Add VSR decision headers if request was successful and didn't hit cache
	if isSuccessful && !ctx.VSRCacheHit && ctx != nil {
		var setHeaders []*core.HeaderValueOption

		// Add x-vsr-selected-category header (from domain classification)
		if ctx.VSRSelectedCategory != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRSelectedCategory,
					RawValue: []byte(ctx.VSRSelectedCategory),
				},
			})
		}

		// Add x-vsr-selected-decision header (from decision evaluation)
		if ctx.VSRSelectedDecisionName != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRSelectedDecision,
					RawValue: []byte(ctx.VSRSelectedDecisionName),
				},
			})
		}

		// Add x-vsr-matched-keywords header (from keyword classification)
		if len(ctx.VSRMatchedKeywords) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedKeywords,
					RawValue: []byte(strings.Join(ctx.VSRMatchedKeywords, ",")),
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

		// Add signal tracking headers
		if len(ctx.VSRMatchedKeywords) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedKeywords,
					RawValue: []byte(strings.Join(ctx.VSRMatchedKeywords, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedEmbeddings) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedEmbeddings,
					RawValue: []byte(strings.Join(ctx.VSRMatchedEmbeddings, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedDomains) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedDomains,
					RawValue: []byte(strings.Join(ctx.VSRMatchedDomains, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedFactCheck) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedFactCheck,
					RawValue: []byte(strings.Join(ctx.VSRMatchedFactCheck, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedUserFeedback) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedUserFeedback,
					RawValue: []byte(strings.Join(ctx.VSRMatchedUserFeedback, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedPreference) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedPreference,
					RawValue: []byte(strings.Join(ctx.VSRMatchedPreference, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedLanguage) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedLanguage,
					RawValue: []byte(strings.Join(ctx.VSRMatchedLanguage, ",")),
				},
			})
		}

		if len(ctx.VSRMatchedLatency) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedLatency,
					RawValue: []byte(strings.Join(ctx.VSRMatchedLatency, ",")),
				},
			})
		}

		// Add x-vsr-matched-context header (from context signal classification)
		if len(ctx.VSRMatchedContext) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedContext,
					RawValue: []byte(strings.Join(ctx.VSRMatchedContext, ",")),
				},
			})
		}

		// Add x-vsr-context-token-count header
		if ctx.VSRContextTokenCount > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRContextTokenCount,
					RawValue: []byte(strconv.Itoa(ctx.VSRContextTokenCount)),
				},
			})
		}

		// Add x-vsr-matched-complexity header (from complexity signal classification)
		if len(ctx.VSRMatchedComplexity) > 0 {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.VSRMatchedComplexity,
					RawValue: []byte(strings.Join(ctx.VSRMatchedComplexity, ",")),
				},
			})
		}

		// Attach router replay identifier when available
		if ctx.RouterReplayID != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.RouterReplayID,
					RawValue: []byte(ctx.RouterReplayID),
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
