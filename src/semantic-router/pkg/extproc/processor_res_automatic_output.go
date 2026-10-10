package extproc

import (
	"strconv"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// bindAutomaticOutputResponseHeaders exposes the final dispatch budget before
// either a buffered or streaming body. Provider headers cannot supply these
// router-owned facts, including when this request did not resolve auto output.
func bindAutomaticOutputResponseHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	common := response.GetResponseHeaders().GetResponse()
	if common == nil {
		return
	}
	mutation := &ext_proc.HeaderMutation{RemoveHeaders: []string{
		headers.VSREffectiveInputTokens, headers.VSREffectiveMaxOutputTokens,
	}}
	defer func() { common.HeaderMutation = mergeHeaderMutations(common.HeaderMutation, mutation) }()
	if ctx == nil || ctx.SkipProcessing || ctx.LooperRequest || ctx.VSRCacheHit ||
		ctx.UpstreamStatusCode < 200 || ctx.UpstreamStatusCode >= 300 || ctx.SemanticRequest == nil {
		return
	}
	sampling := ctx.SemanticRequest.Sampling
	if !sampling.AutomaticOutput || sampling.AutomaticInputTokens == nil || sampling.MaxOutputTokens == nil ||
		*sampling.AutomaticInputTokens <= 0 || *sampling.MaxOutputTokens <= 0 {
		return
	}
	mutation.SetHeaders = []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key: headers.VSREffectiveInputTokens, RawValue: []byte(strconv.FormatInt(*sampling.AutomaticInputTokens, 10)),
			},
			AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
		},
		{
			Header: &core.HeaderValue{
				Key: headers.VSREffectiveMaxOutputTokens, RawValue: []byte(strconv.FormatInt(*sampling.MaxOutputTokens, 10)),
			},
			AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
		},
	}
}
