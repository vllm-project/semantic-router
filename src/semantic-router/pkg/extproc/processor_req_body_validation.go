package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/gjson"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

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
