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

	// Capture Anthropic-only fields from the raw inbound body (cache_control,
	// top_k, metadata.user_id, multi-block system, image blocks, tool_result
	// is_error/array content) and from the incoming headers (anthropic-version,
	// anthropic-beta). For OpenAI-shape inbound the carrier ends up empty and
	// the rebuild is byte-identical to today. The carrier is also stashed on
	// the request context so the header builder can consume the same values
	// without re-parsing.
	passthrough, ptErr := anthropic.BuildPassthroughFromAnthropicBody(ctx.OriginalRequestBody)
	if ptErr != nil {
		logging.Debugf("Anthropic passthrough capture skipped: %v", ptErr)
	}
	if passthrough != nil {
		passthrough.SetHeadersFromIncoming(ctx.Headers)
	}
	ctx.AnthropicPassthrough = passthrough

	anthropicBody, err := anthropic.ToAnthropicRequestBodyWithPassthrough(openAIRequest, passthrough)
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
	if streaming {
		ctx.AnthropicStream = anthropic.NewStreamState()
	}
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
	backendAddress, backendName, err := r.resolveBackendForModel(ctx, targetModel)
	if err != nil {
		logging.Errorf("Anthropic routing backend resolution failed for model %s: %v", targetModel, err)
		return r.createErrorResponse(500, fmt.Sprintf("Backend resolution error: %v", err))
	}
	if accessKey == "" && backendName != "" {
		if ep, ok := r.Config.GetEndpointByName(backendName); ok && ep.APIKey != "" {
			accessKey = ep.APIKey
		}
	}

	messagesPath := anthropic.AnthropicMessagesPath
	profile, profileErr := r.Config.GetProviderProfileForEndpoint(backendName)
	if profileErr != nil {
		logging.Errorf("Anthropic routing profile resolution failed for backend %s: %v", backendName, profileErr)
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if profile != nil {
		if chatPath, pathErr := profile.ResolveChatPath(); pathErr != nil {
			logging.Errorf("Anthropic routing chat path resolution failed for backend %s: %v", backendName, pathErr)
			return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
		} else if chatPath != "" {
			messagesPath = chatPath
			logging.Infof("Anthropic upstream path for model %s: %s", targetModel, messagesPath)
		}
	}

	bodyLength := len(anthropicBody)
	var anthropicHeaders []anthropic.HeaderKeyValue
	if ctx.ExpectStreamingResponse {
		anthropicHeaders = anthropic.BuildStreamingRequestHeadersWithPassthrough(accessKey, bodyLength, messagesPath, ctx.AnthropicPassthrough)
	} else {
		anthropicHeaders = anthropic.BuildRequestHeadersWithPassthrough(accessKey, bodyLength, messagesPath, ctx.AnthropicPassthrough)
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
	appendRoutingHeaders(&setHeaders, targetModel)
	setHeaders = append(setHeaders, r.startUpstreamSpanAndInjectHeaders(targetModel, backendAddress, ctx)...)
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
