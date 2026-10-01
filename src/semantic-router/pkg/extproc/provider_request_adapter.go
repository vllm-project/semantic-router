package extproc

import (
	"slices"
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
	encoded := body
	body, mutation, err := r.projectProviderRequest(body, dispatch, ctx)
	if err != nil {
		return body, err
	}
	if explicitAnthropicReasoningDisabled(ctx, dispatch) &&
		(mutation == nil || !mutation.reasoningApplied) {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
			"the selected backend has no effective reasoning-off control", nil)
	}
	if mutation != nil {
		r.observeReasoningMutation(mutation, dispatch.useReasoning && !explicitAnthropicReasoningDisabled(ctx, dispatch))
	}
	if dispatch != nil {
		body, err = adaptOllamaOutputLimit(body, dispatch.targetFormat, dispatch.profile)
		if err != nil {
			return body, err
		}
	}
	reportDroppedReasoningSummary(ctx, dispatch, encoded, body)
	return body, nil
}

// reportDroppedReasoningSummary records a Responses summary request removed by
// provider adaptation. A chat_template_kwargs backend has no summary control,
// and fallback can adapt the same request more than once.
func reportDroppedReasoningSummary(ctx *RequestContext, dispatch *providerDispatch, encoded, adapted []byte) {
	if dispatch == nil || dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 ||
		resolveProviderReasoningTransport(dispatch.profile) != modelcatalog.ReasoningTransportChatTemplate ||
		ctx == nil || ctx.SemanticRequest == nil || ctx.SemanticRequest.ReasoningSummary == "" ||
		!gjson.GetBytes(encoded, "reasoning.summary").Exists() ||
		gjson.GetBytes(adapted, "reasoning.summary").Exists() {
		return
	}
	diagnostic := llmprotocol.Diagnostic{
		Source: ctx.SourceFormat, Field: "reasoning.summary", Action: llmprotocol.DiagnosticDropped,
		Reason: "chat_template_kwargs cannot request a reasoning summary",
	}
	if !slices.Contains(ctx.ProtocolDiagnostics, diagnostic) {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostic)
	}
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
	if dispatch == nil || ctx == nil {
		return body, nil, nil
	}
	explicitDisable := explicitAnthropicReasoningDisabled(ctx, dispatch)
	if dispatch.decisionName == "" && !explicitDisable {
		return body, nil, nil
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 && !explicitDisable {
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
		dispatch.useReasoning && !explicitDisable,
		ctx.decisionForCandidate(dispatch.logicalModel),
		dispatch.profile,
	)
}
