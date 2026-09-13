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
	if dispatch == nil || ctx == nil || dispatch.decisionName == "" {
		return body, nil
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		family := r.getModelReasoningFamily(dispatch.logicalModel)
		transport := resolveProviderReasoningTransport(dispatch.profile)
		if dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 || family == nil ||
			transport != modelcatalog.ReasoningTransportChatTemplate {
			return body, nil
		}
	}
	return r.setReasoningModeToRequestBodyForModelAndProvider(
		body,
		dispatch.logicalModel,
		dispatch.useReasoning,
		ctx.VSRSelectedDecision,
		dispatch.profile,
	)
}
