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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

type responseHeaderOutcome struct {
	statusCode   int
	isSuccessful bool
}

func (r *OpenAIRouter) handleLooperResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.LooperRequest {
		return nil
	}

	statusCode := 200
	if v != nil && v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
		statusCode = getStatusFromHeaders(v.ResponseHeaders.Headers)
	}

	r.updateRouterReplayStatus(ctx, statusCode, false)
	return buildResponseHeadersContinueResponse(nil, false)
}

func evaluateResponseHeaderOutcome(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) responseHeaderOutcome {
	outcome := responseHeaderOutcome{}
	if v == nil || v.ResponseHeaders == nil || v.ResponseHeaders.Headers == nil {
		return outcome
	}

	if ctx != nil {
		ctx.IsStreamingResponse = isStreamingContentType(v.ResponseHeaders.Headers)
	}

	outcome.statusCode = getStatusFromHeaders(v.ResponseHeaders.Headers)
	outcome.isSuccessful = outcome.statusCode >= 200 && outcome.statusCode < 300
	recordResponseHeaderErrorMetrics(ctx, outcome.statusCode)
	return outcome
}

func recordResponseHeaderErrorMetrics(ctx *RequestContext, statusCode int) {
	if statusCode >= 500 {
		metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_5xx")
		return
	}
	if statusCode >= 400 {
		metrics.RecordRequestError(getModelFromCtx(ctx), "upstream_4xx")
	}
}

func finishUpstreamResponseSpan(ctx *RequestContext, outcome responseHeaderOutcome) {
	if ctx == nil || ctx.UpstreamSpan == nil {
		return
	}

	tracing.SetSpanAttributes(ctx.UpstreamSpan, attribute.Int("http.status_code", outcome.statusCode))
	if !outcome.isSuccessful && outcome.statusCode != 0 {
		ctx.UpstreamSpan.SetStatus(codes.Error, "upstream request failed")
	}

	ctx.UpstreamSpan.End()
	ctx.UpstreamSpan = nil
}

func maybeRecordResponseHeaderTTFT(ctx *RequestContext) {
	if ctx == nil ||
		ctx.IsStreamingResponse ||
		ctx.TTFTRecorded ||
		ctx.ProcessingStartTime.IsZero() ||
		ctx.RequestModel == "" {
		return
	}

	ttft := time.Since(ctx.ProcessingStartTime).Seconds()
	if ttft <= 0 {
		return
	}

	metrics.RecordModelTTFT(ctx.RequestModel, ttft)
	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	latency.UpdateTTFT(ctx.RequestModel, ttft)
	ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
		Model:       ctx.RequestModel,
		TTFTSeconds: ttft,
	})
	maybeEmitTransitionEvent(ctx)
}

func buildResponseHeadersContinueResponse(
	headerMutation *ext_proc.HeaderMutation,
	streaming bool,
) *ext_proc.ProcessingResponse {
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
	if streaming {
		response.ModeOverride = &http_ext.ProcessingMode{
			ResponseBodyMode: http_ext.ProcessingMode_STREAMED,
		}
	}
	return response
}

// getStatusFromHeaders extracts :status pseudo-header value as integer.
func getStatusFromHeaders(headerMap *core.HeaderMap) int {
	if headerMap == nil {
		return 0
	}
	for _, hv := range headerMap.Headers {
		if hv.Key != ":status" {
			continue
		}
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
	return 0
}

func getModelFromCtx(ctx *RequestContext) string {
	if ctx == nil || ctx.RequestModel == "" {
		return "unknown"
	}
	return ctx.RequestModel
}

// isStreamingContentType checks if the response content-type indicates streaming (SSE).
func isStreamingContentType(headerMap *core.HeaderMap) bool {
	if headerMap == nil {
		return false
	}
	for _, hv := range headerMap.Headers {
		if strings.ToLower(hv.Key) != "content-type" {
			continue
		}
		value := extractHeaderValue(hv)
		if strings.Contains(strings.ToLower(value), "text/event-stream") {
			return true
		}
	}
	return false
}
