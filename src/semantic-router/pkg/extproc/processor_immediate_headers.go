package extproc

import (
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

func appendRecipeHeaderToImmediateResponse(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if ctx == nil {
		return
	}
	appendImmediateResponseHeader(response, headers.VSRSelectedRecipe, string(ctx.Routing.RecipeName()))
}
