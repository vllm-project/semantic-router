package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	r.startRouterReplay(ctx, originalModel, targetModel, decisionName)
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
	streaming := ctx.ExpectStreamingResponse
	anthropicBody, err := anthropic.ToAnthropicRequestBody(openAIRequest)
	if err != nil {
		logging.Errorf("Failed to transform request to Anthropic format: %v", err)
		return "", nil, r.createErrorResponse(500, fmt.Sprintf("Request transformation error: %v", err))
	}
	if streaming {
		anthropicBody, err = anthropic.WithStreamingRequestBody(anthropicBody)
		if err != nil {
			logging.Errorf("Failed to enable Anthropic streaming on request: %v", err)
			return "", nil, r.createErrorResponse(500, fmt.Sprintf("Request transformation error: %v", err))
		}
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
	endpoint, endpointName, err := r.selectEndpointForModel(ctx, targetModel)
	if err != nil {
		logging.Errorf("Anthropic routing endpoint selection failed for model %s: %v", targetModel, err)
		return r.createErrorResponse(500, fmt.Sprintf("Endpoint selection error: %v", err))
	}
	if accessKey == "" && endpointName != "" {
		if ep, ok := r.Config.GetEndpointByName(endpointName); ok && ep.APIKey != "" {
			accessKey = ep.APIKey
		}
	}

	messagesPath := anthropic.AnthropicMessagesPath
	profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName)
	if profileErr != nil {
		logging.Errorf("Anthropic routing profile resolution failed for endpoint %s: %v", endpointName, profileErr)
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if profile != nil {
		if chatPath, pathErr := profile.ResolveChatPath(); pathErr != nil {
			logging.Errorf("Anthropic routing chat path resolution failed for endpoint %s: %v", endpointName, pathErr)
			return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
		} else if chatPath != "" {
			messagesPath = chatPath
			logging.Infof("Anthropic upstream path for model %s: %s", targetModel, messagesPath)
		}
	}

	bodyLength := len(anthropicBody)
	var anthropicHeaders []anthropic.HeaderKeyValue
	if ctx.ExpectStreamingResponse {
		anthropicHeaders = anthropic.BuildStreamingRequestHeaders(accessKey, bodyLength, messagesPath)
	} else {
		anthropicHeaders = anthropic.BuildRequestHeaders(accessKey, bodyLength, messagesPath)
	}
	setHeaders := make([]*core.HeaderValueOption, 0, len(anthropicHeaders)+8)
	for _, header := range anthropicHeaders {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      header.Key,
				RawValue: []byte(header.Value),
			},
		})
	}
	appendProfileHeaders(&setHeaders, profile)
	appendRoutingHeaders(&setHeaders, targetModel, endpoint)
	setHeaders = append(setHeaders, r.startUpstreamSpanAndInjectHeaders(targetModel, endpoint, ctx)...)
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
