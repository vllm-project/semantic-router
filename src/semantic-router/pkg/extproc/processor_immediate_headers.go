package extproc

import (
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// appendImmediateResponseHeader is the transport-neutral seam for metadata on
// Router-generated responses. Producers supply Router-owned semantic values;
// this helper owns only the Envoy header mutation representation.
func appendImmediateResponseHeader(response *ext_proc.ProcessingResponse, key, value string) {
	if response == nil || key == "" || value == "" {
		return
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	immediate.Headers.SetHeaders = append(immediate.Headers.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{Key: key, RawValue: []byte(value)},
	})
}

// appendConfigVersionToImmediateResponse names the configuration snapshot
// version that produced a Router-generated response.
func appendConfigVersionToImmediateResponse(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if ctx == nil || ctx.ConfigVersion == 0 {
		return
	}
	appendImmediateResponseHeader(response, headers.VSRConfigVersion, strconv.FormatUint(ctx.ConfigVersion, 10))
}

// appendRoutedResponseHeaders completes a routed response the Router answers
// itself, one with a response path, such as a cache hit, a fast response, a
// Looper answer or a refusal: it adds the public set the request recorded
// and, for a caller allowed the debug headers, x-vsr-trace. A header the
// response already carries is kept.
func appendRoutedResponseHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	immediate := response.GetImmediateResponse()
	if immediate == nil || ctx == nil {
		return
	}
	builder := newResponseHeaderMutationBuilder()
	routed := false
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		key := strings.ToLower(option.GetHeader().GetKey())
		builder.seen[key] = struct{}{}
		routed = routed || key == headers.VSRResponsePath
	}
	if !routed {
		return
	}
	builder.addPublicRoutingHeaders(ctx)
	builder.addWarnings(ctx)
	builder.addTrace(ctx)
	if len(builder.setHeaders) == 0 {
		return
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	immediate.Headers.SetHeaders = append(immediate.Headers.SetHeaders, builder.setHeaders...)
}
