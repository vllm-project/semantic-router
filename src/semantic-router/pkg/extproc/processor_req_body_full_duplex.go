package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// fullDuplexHeaderHold is a request header reply held under
// FULL_DUPLEX_STREAMED until the body is routed. Envoy keeps the request
// headers paused until this reply arrives and ignores header mutations on body
// replies, so the body stage's routing mutations travel on it instead.
type fullDuplexHeaderHold struct {
	response        *ext_proc.ProcessingResponse
	routeMutation   *ext_proc.HeaderMutation
	clearRouteCache bool
}

// holdFullDuplexHeaderReply holds the header reply of a full-duplex request
// whose body the Router accumulates and routes at end of stream.
func (r *OpenAIRouter) holdFullDuplexHeaderReply(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) bool {
	if !ctx.FullDuplexRequestBody || ctx.SkipProcessing || v.RequestHeaders.GetEndOfStream() ||
		response.GetRequestHeaders() == nil || r.Config == nil || !r.Config.StreamedBodyMode {
		return false
	}
	ctx.fullDuplexHold = &fullDuplexHeaderHold{response: response}
	return true
}

// flushHeldRequestHeaderReply sends a held header reply, carrying the body
// stage's routing mutations, before the first body or trailers reply. An
// immediate response replaces it.
func flushHeldRequestHeaderReply(
	stream ext_proc.ExternalProcessor_ProcessServer,
	next *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) error {
	hold := ctx.fullDuplexHold
	if hold == nil {
		return nil
	}
	ctx.fullDuplexHold = nil
	if next.GetImmediateResponse() != nil {
		return nil
	}
	headers := hold.response.GetRequestHeaders()
	if headers.Response == nil {
		headers.Response = &ext_proc.CommonResponse{}
	}
	if hold.routeMutation != nil {
		headers.Response.HeaderMutation = sequenceHeaderMutations(headers.Response.HeaderMutation, hold.routeMutation)
	}
	headers.Response.ClearRouteCache = headers.Response.ClearRouteCache || hold.clearRouteCache
	return sendResponse(stream, hold.response, "request header")
}

// sendHeldHeaderReplyBeforeError sends a held header reply before the Router
// ends the stream with an error, so the header stage's own mutations apply
// whatever the gateway's failure policy does next, as when the reply went out
// at once. Process calls it after recovering a panic, so it recovers its own.
func sendHeldHeaderReplyBeforeError(stream ext_proc.ExternalProcessor_ProcessServer, ctx *RequestContext) {
	defer func() {
		if rec := recover(); rec != nil {
			logging.Errorf("Process: recovered panic sending the held header reply: %v", rec)
		}
	}()
	_ = flushHeldRequestHeaderReply(stream, nil, ctx)
}

// processRequestTrailers handles FULL_DUPLEX_STREAMED request trailers. Envoy
// sends them instead of an end_of_stream body chunk when the request has
// trailers, so they complete the body. Replies keep the headers-body-trailers
// order.
func (r *OpenAIRouter) processRequestTrailers(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
) error {
	body := ctx.StreamedBody
	if body == nil && ctx.fullDuplexHold != nil {
		// The request ended without a body chunk (HTTP/2 headers, then trailers).
		// The body stage rejects its empty body, which never decodes, instead of
		// the request going out on the client's own route.
		body = newStreamedBodyHandler(r, ctx)
	}
	if body != nil {
		ctx.StreamedBody = nil
		response, err := body.finishAtTrailers()
		body.Release()
		immediate, err := r.sendRequestBodyResult(stream, response, err, ctx)
		if err != nil || immediate {
			return err
		}
	}
	return sendResponse(stream, &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestTrailers{RequestTrailers: &ext_proc.TrailersResponse{}},
	}, "request trailers")
}

// sequenceHeaderMutations returns one mutation with the effect of applying
// earlier, then later. Envoy runs every remove before any set, so an earlier set
// of a header that the later mutation removes is dropped.
func sequenceHeaderMutations(earlier, later *ext_proc.HeaderMutation) *ext_proc.HeaderMutation {
	removed := make(map[string]bool, len(later.GetRemoveHeaders()))
	for _, name := range later.GetRemoveHeaders() {
		removed[strings.ToLower(name)] = true
	}
	merged := &ext_proc.HeaderMutation{}
	for _, set := range earlier.GetSetHeaders() {
		if !removed[strings.ToLower(set.GetHeader().GetKey())] {
			merged.SetHeaders = append(merged.SetHeaders, set)
		}
	}
	merged.SetHeaders = append(merged.SetHeaders, later.GetSetHeaders()...)
	merged.RemoveHeaders = append(append(merged.RemoveHeaders, earlier.GetRemoveHeaders()...), later.GetRemoveHeaders()...)
	return merged
}
