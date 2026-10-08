package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Each phase's reply is built here once and shared by every adapter: the
// ext_proc stream sends it to Envoy, and a routing session hands its effect to
// the native gateway. The *Sent functions run what follows a delivered reply.

// requestHeadersReply runs the request-headers phase.
func (r *OpenAIRouter) requestHeadersReply(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	response, err := r.handleRequestHeaders(v, ctx)
	if err != nil {
		logging.Errorf("handleRequestHeaders failed: %v", err)
		return nil, err
	}
	response = r.encodeImmediateResponseForClient(response, ctx)
	appendConfigVersionToImmediateResponse(response, ctx)
	r.bindBenchmarkConfigResponse(response, ctx)
	return response, nil
}

// requestBodyReply turns the request-body phase result into its reply. A
// client-owned protocol error becomes an immediate response; any other error
// ends the request.
func (r *OpenAIRouter) requestBodyReply(
	response *ext_proc.ProcessingResponse,
	err error,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if err != nil {
		var ok bool
		if response, ok = r.processBodyRoutingError(err, ctx); !ok {
			logging.Errorf("handleRequestBody failed: %v", err)
			return nil, err
		}
	}
	response = r.encodeImmediateResponseForClient(response, ctx)
	appendConfigVersionToImmediateResponse(response, ctx)
	r.bindBenchmarkConfigResponse(response, ctx)
	r.persistImmediateResponseObject(response, ctx)
	return response, nil
}

// responseHeadersReply runs the response-headers phase.
func (r *OpenAIRouter) responseHeadersReply(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	response, err := r.handleResponseHeaders(v, ctx)
	if err != nil {
		return nil, err
	}
	appendConfigVersionToImmediateResponse(response, ctx)
	r.bindBenchmarkConfigResponse(response, ctx)
	return response, nil
}

// responseBodyReply runs the response-body phase for one buffered body or
// streamed chunk.
func (r *OpenAIRouter) responseBodyReply(
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	response, err := r.handleResponseBody(v, ctx)
	if err != nil {
		return nil, err
	}
	appendConfigVersionToImmediateResponse(response, ctx)
	r.bindBenchmarkConfigResponse(response, ctx)
	return response, nil
}

func responseHeadersReplySent(ctx *RequestContext, response *ext_proc.ProcessingResponse, endOfStream bool) {
	finishImmediateResponseTrace(ctx, response)
	if endOfStream {
		finishRequestTrace(ctx, nil)
	}
}

func responseBodyReplySent(ctx *RequestContext, response *ext_proc.ProcessingResponse, endOfStream bool) {
	finishImmediateResponseTrace(ctx, response)
	if endOfStream || ctx.StreamingComplete {
		finishRequestTrace(ctx, nil)
	}
}
