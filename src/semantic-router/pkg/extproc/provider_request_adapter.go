package extproc

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ollamaProviderType is the catalog provider id for Ollama. Ollama's OpenAI
// layer reads the output limit only from max_tokens, and documents no
// max_completion_tokens support:
// https://github.com/ollama/ollama/blob/6383a0fa9cbf97494b847226e189f6e36b401a08/docs/api/openai-compatibility.mdx?plain=1#L224-L243
const ollamaProviderType = "ollama"

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
	if err != nil {
		return body, err
	}
	if mutation != nil {
		r.observeReasoningMutation(mutation, dispatch.useReasoning)
	}
	if dispatch == nil {
		return body, nil
	}
	return adaptOllamaOutputLimit(body, dispatch.targetFormat, dispatch.profile)
}

// adaptOllamaOutputLimit moves the Chat codec's max_completion_tokens to the
// max_tokens field Ollama reads, so the client's limit still caps generation.
func adaptOllamaOutputLimit(body []byte, format llmprotocol.WireFormat, profile *config.ProviderProfile) ([]byte, error) {
	if format != llmprotocol.OpenAIChatV1 || profile == nil ||
		!strings.EqualFold(strings.TrimSpace(profile.Type), ollamaProviderType) {
		return body, nil
	}
	limit := gjson.GetBytes(body, "max_completion_tokens")
	if !limit.Exists() {
		return body, nil
	}
	body, err := sjson.SetRawBytes(body, "max_tokens", []byte(limit.Raw))
	if err != nil {
		return nil, err
	}
	return sjson.DeleteBytes(body, "max_completion_tokens")
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
