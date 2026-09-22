package extproc

import (
	"encoding/json"
	"strconv"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func automaticBudgetResponseHeaders(status, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: status},
			{Key: "content-type", Value: contentType},
			{Key: headers.VSREffectiveInputTokens, Value: "999"},
			{Key: headers.VSREffectiveMaxOutputTokens, Value: "999"},
		}},
	}}
}

func TestAutomaticOutputResponseHeadersMatchFinalSelectedDispatch(t *testing.T) {
	for _, window := range []int{32768, 65536} {
		for _, contentType := range []string{"application/json", "text/event-stream"} {
			t.Run(strconv.Itoa(window)+"/"+contentType, func(t *testing.T) {
				calls := 0
				router, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
				model := ctx.VSRSelectedDecision.ModelRefs[0].Model
				if window == 65536 {
					params := router.Config.ModelConfig[model]
					params.ContextWindowSize = window
					params.MaxOutputTokens = window
					params.ExternalModelIDs = nil
					model = "wide"
					router.Config.ModelConfig[model] = params
					ctx.VSRSelectedDecision.ModelRefs = []config.ModelRef{{Model: model}}
				}
				ctx.SemanticRequest.Stream = contentType == "text/event-stream"
				require.NoError(t, router.prepareDecisionContextOverflow(ctx, "auto"))
				_, err := router.decisionEligibleModelRefs(ctx.VSRSelectedDecision, ctx)
				require.NoError(t, err)
				candidateInput := ctx.AutomaticCandidateDemands[model].InputTokens
				dispatch, err := router.prepareProviderDispatch(ctx.SemanticRequest, model, ctx.VSRSelectedDecision.Name, false, ctx)
				require.NoError(t, err)
				// Late request changes require another render; headers must describe
				// that final wire request, not the selection-time candidate demand.
				ctx.SemanticRequest.Messages[0].Content[0].Text += " revised"
				ctx.SemanticRequest.Generation++
				requestResponse := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}},
				}}
				requestResponse, err = router.finalizeProviderDispatchResponse(dispatch, requestResponse, ctx)
				require.NoError(t, err)
				var wire map[string]any
				require.NoError(t, json.Unmarshal(requestResponse.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &wire))
				wireLimit := wire["max_tokens"]
				if wireLimit == nil {
					wireLimit = wire["max_completion_tokens"]
				}
				input := candidateInput + len(" revised")
				require.EqualValues(t, window-input, wireLimit)
				require.Equal(t, dispatch.upstreamModel, wire["model"])

				response, err := router.handleResponseHeaders(automaticBudgetResponseHeaders("200", contentType), ctx)
				require.NoError(t, err)
				mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
				require.Equal(t, strconv.Itoa(input), headerValueForTest(mutation, headers.VSREffectiveInputTokens))
				require.Equal(t, strconv.Itoa(window-input), headerValueForTest(mutation, headers.VSREffectiveMaxOutputTokens))
				require.Equal(t, model, headerValueForTest(mutation, headers.VSRSelectedModel))
				require.Equal(t, contentType == "text/event-stream", ctx.IsStreamingResponse)
				for _, key := range []string{headers.VSREffectiveInputTokens, headers.VSREffectiveMaxOutputTokens} {
					require.Equal(t, core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD, headerAppendActionForTest(mutation, key))
					require.Contains(t, mutation.RemoveHeaders, key)
				}
			})
		}
	}
}

func TestAutomaticOutputResponseHeadersAbsentWithoutResolvedDispatch(t *testing.T) {
	for _, name := range []string{"explicit", "unresolved", "missing-output", "zero-input", "negative-output", "cache", "skip", "looper", "upstream-error", "no-request"} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{Sampling: llmprotocol.Sampling{
				AutomaticOutput: true, AutomaticInputTokens: llmprotocol.Int64(12), MaxOutputTokens: llmprotocol.Int64(100),
			}}}
			status := "200"
			switch name {
			case "explicit":
				ctx.SemanticRequest.Sampling.AutomaticOutput = false
			case "unresolved":
				ctx.SemanticRequest.Sampling.AutomaticInputTokens = nil
			case "missing-output":
				ctx.SemanticRequest.Sampling.MaxOutputTokens = nil
			case "zero-input":
				ctx.SemanticRequest.Sampling.AutomaticInputTokens = llmprotocol.Int64(0)
			case "negative-output":
				ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(-1)
			case "cache":
				ctx.VSRCacheHit = true
			case "skip":
				ctx.SkipProcessing = true
			case "looper":
				ctx.LooperRequest = true
			case "upstream-error":
				status = "503"
			case "no-request":
				ctx.SemanticRequest = nil
			}
			response, err := (&OpenAIRouter{}).handleResponseHeaders(automaticBudgetResponseHeaders(status, "application/json"), ctx)
			require.NoError(t, err)
			mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
			for _, key := range []string{headers.VSREffectiveInputTokens, headers.VSREffectiveMaxOutputTokens} {
				require.Empty(t, headerValueForTest(mutation, key))
				require.Contains(t, mutation.GetRemoveHeaders(), key)
			}
		})
	}
}
