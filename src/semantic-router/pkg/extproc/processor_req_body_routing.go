package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// routeHeaderState is intentionally provider-neutral. ExtProc may emit
// logical routing metadata and route-local plugin mutations, but physical
// endpoint, credential, protocol, retry, and fallback decisions belong to the
// immutable BackendInvoker plan.
type routeHeaderState struct {
	setHeaders    []*core.HeaderValueOption
	removeHeaders []string
}

func (r *OpenAIRouter) modifyRequestBodyForEntrypointRouting(
	request *llmprotocol.Request,
	matchedModel string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) ([]byte, error) {
	if request == nil {
		return nil, status.Error(codes.Internal, "neutral inference request is unavailable")
	}
	changed := request.Model != matchedModel || request.Stream != ctx.ExpectStreamingResponse
	request.Model = matchedModel
	request.Stream = ctx.ExpectStreamingResponse

	if decisionName != "" {
		changed = r.applySemanticReasoningMode(request, useReasoning, ctx.VSRSelectedDecision) || changed
		injected, err := r.addSemanticSystemPromptIfConfigured(request, decisionName, matchedModel, ctx)
		if err != nil {
			return nil, err
		}
		changed = injected || changed
	}

	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.GetRequestParamsConfig() != nil {
		paramsChanged, err := r.applySemanticRequestParams(
			ctx.VSRSelectedDecision,
			request,
			ctx.Routing.RuntimeScope(),
		)
		if err != nil {
			return nil, err
		}
		changed = paramsChanged || changed
	}
	if changed {
		request.Generation++
	}
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		logging.Errorf("Error encoding neutral request: %v", err)
		metrics.RecordRequestError(matchedModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error encoding inference request: %v", err)
	}
	return body, nil
}

func appendContentLengthHeader(setHeaders *[]*core.HeaderValueOption, bodyLength int) {
	*setHeaders = append(*setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      "content-length",
			RawValue: []byte(fmt.Sprintf("%d", bodyLength)),
		},
	})
}

func (r *OpenAIRouter) applyDecisionHeaderMutations(state *routeHeaderState, ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	pluginSetHeaders, pluginRemoveHeaders := r.buildHeaderMutations(ctx.VSRSelectedDecision)
	state.setHeaders = append(state.setHeaders, pluginSetHeaders...)
	state.removeHeaders = append(state.removeHeaders, pluginRemoveHeaders...)
}

func buildRequestBodyContinueResponse(
	state *routeHeaderState,
	bodyMutation *ext_proc.BodyMutation,
	clearRouteCache bool,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:          ext_proc.CommonResponse_CONTINUE,
					ClearRouteCache: clearRouteCache,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    state.setHeaders,
						RemoveHeaders: state.removeHeaders,
					},
					BodyMutation: bodyMutation,
				},
			},
		},
	}
}

// getModelParams returns model params for looper/model helpers.
func (r *OpenAIRouter) getModelParams() map[string]config.ModelParams {
	if r.Config == nil || r.Config.ModelConfig == nil {
		return nil
	}
	return r.Config.ModelConfig
}
