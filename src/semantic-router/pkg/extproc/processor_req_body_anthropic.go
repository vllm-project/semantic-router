package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleAnthropicRouting handles routing to Anthropic Claude API via Envoy.
// Transforms the request body from OpenAI format to Anthropic format and sets
// appropriate headers for Envoy to route to the Anthropic cluster.
func (r *OpenAIRouter) handleAnthropicRouting(
	openAIRequest *openai.ChatCompletionNewParams,
	originalModel string,
	targetModel string,
	decisionName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Routing to Anthropic API via Envoy for model: %s (original: %s)", targetModel, originalModel)
	if ctx.ExpectStreamingResponse {
		logging.Warnf("Streaming not supported for Anthropic backend, rejecting request for model: %s", targetModel)
		return r.createErrorResponse(
			400,
			"Streaming is not supported for Anthropic models. Please set stream=false in your request.",
		), nil
	}

	accessKey, anthropicBody, errorResponse := r.prepareAnthropicRoutingRequest(
		openAIRequest,
		targetModel,
		decisionName,
		ctx,
	)
	if errorResponse != nil {
		return errorResponse, nil
	}
	logging.Infof("Transformed request for Anthropic API, body size: %d bytes", len(anthropicBody))
	return r.buildAnthropicRoutingResponse(targetModel, accessKey, anthropicBody, ctx), nil
}

func (r *OpenAIRouter) prepareAnthropicRoutingRequest(
	openAIRequest *openai.ChatCompletionNewParams,
	targetModel string,
	decisionName string,
	ctx *RequestContext,
) (string, []byte, *ext_proc.ProcessingResponse) {
	accessKey, err := r.CredentialResolver.KeyForProvider(authz.ProviderAnthropic, targetModel, ctx.Headers)
	if err != nil {
		return "", nil, r.createErrorResponse(
			401,
			fmt.Sprintf("Credential resolution failed for model %s: %v", targetModel, err),
		)
	}
	if accessKey == "" {
		logging.Debugf("No API key for Anthropic model %q (fail_open=true) — request will use empty key", targetModel)
	}

	openAIRequest.Model = targetModel
	anthropicBody, err := anthropic.ToAnthropicRequestBody(openAIRequest)
	if err != nil {
		logging.Errorf("Failed to transform request to Anthropic format: %v", err)
		return "", nil, r.createErrorResponse(500, fmt.Sprintf("Request transformation error: %v", err))
	}

	ctx.RequestModel = targetModel
	ctx.VSRSelectedModel = targetModel
	ctx.APIFormat = config.APIFormatAnthropic
	if decisionName != "" {
		ctx.VSRSelectedDecision = r.Config.GetDecisionByName(decisionName)
	}

	return accessKey, anthropicBody, nil
}

func (r *OpenAIRouter) buildAnthropicRoutingResponse(
	targetModel string,
	accessKey string,
	anthropicBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	anthropicHeaders := anthropic.BuildRequestHeaders(accessKey, len(anthropicBody))
	setHeaders := make([]*core.HeaderValueOption, 0, len(anthropicHeaders)+2)
	for _, header := range anthropicHeaders {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      header.Key,
				RawValue: []byte(header.Value),
			},
		})
	}
	setHeaders = append(setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.SelectedModel,
			RawValue: []byte(targetModel),
		},
	})
	setHeaders = append(setHeaders, r.startUpstreamSpanAndInjectHeaders(targetModel, "api.anthropic.com", ctx)...)
	r.recordRoutingLatency(ctx)

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:          ext_proc.CommonResponse_CONTINUE,
					ClearRouteCache: true,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    setHeaders,
						RemoveHeaders: append(anthropic.HeadersToRemove(), r.CredentialResolver.HeadersToStrip()...),
					},
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{
							Body: anthropicBody,
						},
					},
				},
			},
		},
	}
}
