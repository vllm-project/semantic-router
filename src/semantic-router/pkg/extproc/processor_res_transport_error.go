package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func isUpstreamTransportError(ctx *RequestContext) bool {
	return ctx != nil && ctx.UpstreamStatusCode != 0 &&
		(ctx.UpstreamStatusCode < 200 || ctx.UpstreamStatusCode >= 300)
}

// handleUpstreamTransportError is the response-body boundary for HTTP failures.
// A non-2xx body is never decoded as a model response: it is translated through
// the neutral transport-error contract while Envoy preserves the upstream HTTP
// status observed during the response-header phase.
func (r *OpenAIRouter) handleUpstreamTransportError(
	body []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	engine, err := r.protocolEngine()
	if err != nil {
		return r.createErrorResponse(503, "protocol runtime unavailable")
	}
	source, target := responseWireFormats(ctx)
	translated, err := engine.TranslateTransportError(source, target, body, nil)
	if err != nil {
		metrics.RecordRequestError(ctx.RequestModel, "invalid_upstream_error")
		logging.ComponentErrorEvent("extproc", "neutral_transport_error_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     source,
			"status":     ctx.UpstreamStatusCode,
			"error":      err.Error(),
		})
		protocolError := upstreamTransportFallback(ctx.UpstreamStatusCode, err)
		encoded, encodeErr := engine.EncodeError(target, protocolError)
		if encodeErr != nil {
			return r.createErrorResponse(502, "The selected model returned an invalid error response")
		}
		translated.Body = encoded
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, translated.Diagnostics...)
	response := buildResponseBodyContinueResponse(nil, nil)
	setResponseBodyMutation(response, translated.Body)
	setResponseContentType(response, "application/json")
	r.attachRouterReplayResponse(ctx, translated.Body, true)
	return response
}

func responseWireFormats(ctx *RequestContext) (llmprotocol.WireFormat, llmprotocol.WireFormat) {
	source, target := llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	if ctx == nil {
		return source, target
	}
	if ctx.TargetFormat != "" {
		source = ctx.TargetFormat
	} else if ctx.SourceFormat != "" {
		source = ctx.SourceFormat
	}
	if ctx.SourceFormat != "" {
		target = ctx.SourceFormat
	}
	return source, target
}

func upstreamTransportFallback(status int, cause error) *llmprotocol.ProtocolError {
	category, code := llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_error"
	if status == 429 {
		category, code = llmprotocol.ErrorRateLimited, "rate_limited"
	} else if status == 408 || status == 504 {
		category, code = llmprotocol.ErrorUpstreamTimeout, "upstream_timeout"
	}
	return llmprotocol.NewError(category, code, "model service returned an invalid error response", cause)
}
