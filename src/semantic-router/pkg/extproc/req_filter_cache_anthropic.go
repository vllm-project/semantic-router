package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/gjson"

	anthropicapi "github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

func translateAnthropicCacheHit(
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
) *ext_proc.ProcessingResponse {
	if ctx == nil || response == nil {
		return response
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return response
	}
	model := anthropicFastResponseModel(ctx)
	if !ctx.ExpectStreamingResponse {
		body, err := anthropicapi.EmitAnthropicResponse(
			immediate.Body,
			ctx.IRExtensions,
			model,
		)
		if err != nil {
			immediate.Body = anthropicapi.EmitAnthropicError(
				"api_error",
				fmt.Sprintf("cached response replay failed: %v", err),
			)
			return response
		}
		immediate.Body = body
		return response
	}

	body, _, err := anthropicapi.EmitAnthropicSSEChunk(
		immediate.Body,
		anthropicapi.NewStreamState(),
		ctx.IRExtensions,
		model,
	)
	if err != nil || len(body) == 0 {
		immediate.Body = buildAnthropicFastResponseStream(
			"Cached response replay failed.",
			model,
		)
		return response
	}
	immediate.Body = body
	return response
}

func hydrateAnthropicCacheUsage(ctx *RequestContext, cachedBody []byte) {
	if ctx.IRExtensions == nil {
		ctx.IRExtensions = &ir.IRExtensions{
			SourceProtocol: anthropicapi.SourceProtocolAnthropic,
		}
	}
	details := gjson.GetBytes(cachedBody, "usage.prompt_tokens_details")
	if !details.Exists() {
		return
	}
	ctx.IRExtensions.CacheReadInputTokens = details.Get("cached_tokens").Int()
	ctx.IRExtensions.CacheCreationInputTokens = details.Get("cache_write_tokens").Int()
}
