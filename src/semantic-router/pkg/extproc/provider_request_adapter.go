package extproc

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// adaptProviderRequest applies backend-dialect extensions after the standard
// wire codec has rendered the request. Official protocol semantics stay in
// llmprotocol/protocolcodec; model-server extensions such as vLLM
// chat_template_kwargs remain isolated at this final provider boundary.
func (r *OpenAIRouter) adaptProviderRequest(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	body, mutation, err := r.projectProviderRequest(body, dispatch, ctx)
	if err == nil && mutation != nil {
		r.observeReasoningMutation(mutation, dispatch.useReasoning)
	}
	return body, err
}

// projectProviderRequest shares the exact provider dialect with dispatch while
// leaving live reasoning observations to the actual dispatch adapter.
func (r *OpenAIRouter) projectProviderRequest(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, *reasoningRequestMutation, error) {
	if dispatch == nil || ctx == nil || dispatch.decisionName == "" {
		return body, nil, nil
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		family := r.getModelReasoningFamily(dispatch.logicalModel)
		transport := resolveProviderReasoningTransport(dispatch.profile)
		if dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 || family == nil ||
			transport != modelcatalog.ReasoningTransportChatTemplate {
			return body, nil, nil
		}
	}
	return r.projectReasoningRequest(
		body,
		dispatch.logicalModel,
		dispatch.useReasoning,
		ctx.decisionForCandidate(dispatch.logicalModel),
		dispatch.profile,
	)
}
