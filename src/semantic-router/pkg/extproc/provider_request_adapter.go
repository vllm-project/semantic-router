package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// adaptProviderRequest applies backend-dialect extensions after the standard
// wire codec has rendered the request. Official protocol semantics stay in
// llmprotocol/protocolcodec; model-server extensions such as vLLM
// chat_template_kwargs remain isolated at this final provider boundary.
func (r *OpenAIRouter) adaptProviderRequest(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	if dispatch == nil || ctx == nil || dispatch.decisionName == "" || dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		return body, nil
	}
	return r.setReasoningModeToRequestBodyForModelAndProvider(
		body,
		dispatch.logicalModel,
		dispatch.useReasoning,
		ctx.VSRSelectedDecision,
		dispatch.profile,
	)
}
