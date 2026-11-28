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

// maxTokensChatProviders are the catalog provider ids whose Chat Completions
// API takes its output limit as max_tokens and documents no
// max_completion_tokens.
//
// Ollama's OpenAI layer reads the output limit only from max_tokens:
// https://github.com/ollama/ollama/blob/6383a0fa9cbf97494b847226e189f6e36b401a08/docs/api/openai-compatibility.mdx?plain=1#L224-L243
//
// The LongCat API reference lists max_tokens as its only output limit:
// https://longcat.ai/platform/docs/api/chat
var maxTokensChatProviders = []string{"ollama", "longcat"}

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
	enabled := dispatch != nil && dispatch.useReasoning && !explicitAnthropicReasoningDisabled(ctx, dispatch)
	if mutation != nil {
		r.observeReasoningMutation(mutation, enabled)
	}
	if dispatch != nil {
		dispatch.reasoningMode = r.sentReasoningMode(dispatch, mutation, enabled)
		dispatch.reasoningEffort = sentReasoningEffort(body, mutation, enabled)
		body, err = adaptMaxTokensOutputLimit(body, dispatch.targetFormat, dispatch.profile)
		if err != nil {
			return body, err
		}
	}
	reportDroppedReasoningSummary(ctx, dispatch, encoded, body)
	r.reportApproximatedReasoningDisplay(ctx, dispatch)
	return body, nil
}

// sentReasoningMode is on or off where the Router wrote reasoning for the
// dispatched model: the reasoning projection rewrote the request, or the
// decision's controls went to a model with a reasoning family. A request passed
// through with its reasoning as the client sent it, such as one to a model
// without a reasoning family, carries no mode of the Router's.
func (r *OpenAIRouter) sentReasoningMode(dispatch *providerDispatch, mutation *reasoningRequestMutation, enabled bool) string {
	if mutation == nil && (dispatch.decisionName == "" || r.getModelReasoningFamily(dispatch.effectiveBackendModel()) == nil) {
		return ""
	}
	if enabled {
		return "on"
	}
	return "off"
}

// sentReasoningEffort is the reasoning effort a provider request carries:
// the one the reasoning projection wrote when it rewrote the request's
// reasoning, none when it turned reasoning off, and otherwise the effort the
// request was encoded with, which reaches the backend as the client sent it.
func sentReasoningEffort(body []byte, mutation *reasoningRequestMutation, enabled bool) string {
	if mutation != nil {
		if !enabled {
			return ""
		}
		return mutation.appliedEffort
	}
	for _, path := range [...]string{"reasoning_effort", "reasoning.effort", "output_config.effort"} {
		if effort := gjson.GetBytes(body, path); effort.Type == gjson.String {
			return effort.String()
		}
	}
	return ""
}

// reportApproximatedReasoningDisplay records an Anthropic summarized or updates
// display sent to a Chat or Responses backend whose reasoning is on: a model
// with a reasoning family the decision does not turn reasoning off for. The
// Router knows nothing of the reasoning of a model without a family.
func (r *OpenAIRouter) reportApproximatedReasoningDisplay(ctx *RequestContext, dispatch *providerDispatch) {
	if ctx == nil || ctx.SemanticRequest == nil || dispatch == nil ||
		ctx.SourceFormat != llmprotocol.AnthropicMessagesV1 || dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 ||
		(ctx.SemanticRequest.ReasoningDisplay != "summarized" && ctx.SemanticRequest.ReasoningDisplay != "updates") ||
		r.getModelReasoningFamily(dispatch.effectiveBackendModel()) == nil ||
		(dispatch.decisionName != "" && !dispatch.useReasoning) {
		return
	}
	diagnostic := llmprotocol.Diagnostic{
		Source: llmprotocol.AnthropicMessagesV1, Target: dispatch.targetFormat, Field: "thinking.display",
		Action: llmprotocol.DiagnosticApproximated, Reason: "the backend's reasoning is returned as it is, not summarized",
	}
	if !slices.Contains(ctx.ProtocolDiagnostics, diagnostic) {
		ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostic)
	}
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

// adaptMaxTokensOutputLimit moves the Chat codec's max_completion_tokens to
// the max_tokens field these providers read, so the client's limit still caps
// generation.
func adaptMaxTokensOutputLimit(body []byte, format llmprotocol.WireFormat, profile *config.ProviderProfile) ([]byte, error) {
	if format != llmprotocol.OpenAIChatV1 || profile == nil ||
		!slices.Contains(maxTokensChatProviders, strings.ToLower(strings.TrimSpace(profile.Type))) {
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
		family := r.getModelReasoningFamily(dispatch.effectiveBackendModel())
		transport := resolveProviderReasoningTransport(dispatch.profile)
		if dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 || family == nil ||
			transport != modelcatalog.ReasoningTransportChatTemplate {
			return body, nil, nil
		}
	}
	return r.projectReasoningRequest(
		body,
		dispatch.effectiveBackendModel(),
		dispatch.useReasoning && !explicitDisable,
		ctx.decisionForBackend(dispatch.logicalModel),
		dispatch.profile,
	)
}
