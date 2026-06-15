package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/gjson"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) validationResponseFromRequestError(err error) *ext_proc.ProcessingResponse {
	if err == nil {
		return nil
	}

	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		return nil
	}

	return r.createErrorResponse(400, st.Message())
}

func (r *OpenAIRouter) validateRequestBody(requestBody []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if !gjson.ValidBytes(requestBody) {
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		metrics.RecordModelRequest(ctx.RequestModel)
		return r.createErrorResponse(400, "malformed JSON body")
	}

	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		return nil
	}

	if ctx.ClientProtocol == config.ClientProtocolAnthropic {
		return r.validateAnthropicRequestBody(requestBody)
	}

	path := normalizeRequestPath(ctx.Headers[":path"])
	if path != "" && path != "/v1/chat/completions" {
		return nil
	}

	messages := gjson.GetBytes(requestBody, "messages")
	if !messages.Exists() {
		return r.createErrorResponse(400, "messages field is required")
	}
	if !messages.IsArray() {
		return r.createErrorResponse(400, "messages field must be an array")
	}

	return nil
}

// validateAnthropicRequestBody runs the minimum shape checks the
// Anthropic Messages spec mandates before the SDK parser ever sees the
// body. We do not duplicate the parser's per-block validation here —
// just the two top-level fields whose absence makes the rest of the
// pipeline ill-defined.
func (r *OpenAIRouter) validateAnthropicRequestBody(requestBody []byte) *ext_proc.ProcessingResponse {
	model := gjson.GetBytes(requestBody, "model")
	if !model.Exists() || model.Type != gjson.String || model.String() == "" {
		return r.createErrorResponse(400, "model field is required")
	}
	messages := gjson.GetBytes(requestBody, "messages")
	if !messages.Exists() {
		return r.createErrorResponse(400, "messages field is required")
	}
	if !messages.IsArray() {
		return r.createErrorResponse(400, "messages field must be an array")
	}
	return nil
}
