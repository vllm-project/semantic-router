package extproc

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// projectRequestForBackend applies portable request projections before model
// qualification and again before encoding. Only the encoding pass publishes
// diagnostics, so failed candidate probes cannot leak warnings to the client.
func (r *OpenAIRouter) projectRequestForBackend(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
	shadows ...*shadowTarget,
) (llmprotocol.Request, error) {
	projected, _, err := r.projectRequestForBackendWithDiagnostics(request, model, target, shadows...)
	return projected, err
}

func (r *OpenAIRouter) projectRequestForBackendWithDiagnostics(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
	shadows ...*shadowTarget,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	projected, reasoningDiagnostics := r.projectResponsesReasoningForBackend(request, model, target, shadows...)
	projected, budgetDiagnostics, err := r.projectUnsupportedReasoningBudget(projected, model, target)
	if err != nil {
		return projected, nil, err
	}
	projected, verbosityDiagnostics := llmprotocol.ProjectTextVerbosity(projected, target)
	projected, fileDiagnostics := r.projectUnreadableFiles(projected, model, target)
	projected, storeDiagnostics := r.projectUnstoredRequest(projected, target)
	projected, samplingDiagnostics := r.projectUnsupportedSampling(projected, model, target)
	projected, otherDiagnostics, err := r.projectAnthropicRequestForBackendWithDiagnostics(projected, model, target)
	return projected, slices.Concat(reasoningDiagnostics, budgetDiagnostics, verbosityDiagnostics, fileDiagnostics, storeDiagnostics, samplingDiagnostics, otherDiagnostics), err
}

// projectUnsupportedReasoningBudget follows the same projection path during
// candidate qualification and final dispatch. A non-vLLM Chat provider fails
// closed unless the operator explicitly accepts dropping its budget.
func (r *OpenAIRouter) projectUnsupportedReasoningBudget(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	if request.ReasoningBudgetTokens == nil || target != llmprotocol.OpenAIChatV1 || r.providerTypeForModel(model) == vllmProviderType {
		return request, nil, nil
	}
	field := llmprotocol.LossyFieldReasoningBudget
	if r == nil || r.Config == nil || !slices.Contains(r.Config.AllowLossyTranslation, field) {
		return request, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"lossy_translation",
			"translation would lose "+field+llmprotocol.OptInHint(field),
			nil,
		)
	}
	projected, diagnostics := llmprotocol.ProjectUnsupportedReasoningBudget(request, target)
	return projected, diagnostics, nil
}

// providerTypeForModel resolves the provider selected by the same primary
// backend lookup used for dispatch. Legacy endpoints can still identify their
// provider through VLLMEndpoint.Type.
func (r *OpenAIRouter) providerTypeForModel(model string) string {
	if r == nil || r.Config == nil {
		return ""
	}
	_, endpointName, found, err := r.Config.ResolvePrimaryBackendForModel(model)
	if err != nil || !found {
		return ""
	}
	if profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName); profileErr == nil && profile != nil {
		if providerType, typeErr := profile.ProviderType(); typeErr == nil {
			return strings.ToLower(strings.TrimSpace(providerType))
		}
		return strings.ToLower(strings.TrimSpace(profile.Type))
	}
	if endpoint, exists := r.Config.GetEndpointByName(endpointName); exists {
		return strings.ToLower(strings.TrimSpace(endpoint.Type))
	}
	return ""
}

// appendRequestProjectionDiagnostics keeps a projection warning to one entry
// when the same request is encoded again during a retry or fallback.
func appendRequestProjectionDiagnostics(ctx *RequestContext, diagnostics llmprotocol.Diagnostics) {
	if ctx == nil {
		return
	}
	for _, diagnostic := range diagnostics {
		if !slices.Contains(ctx.ProtocolDiagnostics, diagnostic) {
			ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostic)
		}
	}
}

// projectUnsupportedSampling sends a model the request without the sampling
// controls its unsupported_capabilities lists, such as min_p for a vLLM
// backend that runs speculative decoding. Selection, dispatch and every
// fallback candidate project through here, so another model still receives
// them.
func (r *OpenAIRouter) projectUnsupportedSampling(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics) {
	if r == nil || r.Config == nil {
		return request, nil
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || len(params.UnsupportedCapabilities) == 0 {
		return request, nil
	}
	unsupported, err := llmprotocol.ParseUnsupportedCapabilities(params.UnsupportedCapabilities)
	if err != nil {
		return request, nil
	}
	return llmprotocol.ProjectUnsupportedSampling(request, unsupported, target)
}

// projectUnstoredRequest sends store: true without it to a backend whose wire
// format keeps no completions, once the operator accepts that loss.
func (r *OpenAIRouter) projectUnstoredRequest(
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics) {
	if r == nil || r.Config == nil || !slices.Contains(r.Config.AllowLossyTranslation, llmprotocol.LossyFieldStore) {
		return request, nil
	}
	if available, ok := r.codecCapabilitiesForFormat(target); ok && available.Supports(llmprotocol.CapabilityRequestStorage) {
		return request, nil
	}
	return llmprotocol.ProjectUnstoredRequest(request, target)
}

// projectUnreadableFiles sends files as notes to a model that declares its
// capabilities without file input, once the operator accepts that loss. Both
// candidate qualification and dispatch project through here, so a model that
// can read the file is never told it was left out.
func (r *OpenAIRouter) projectUnreadableFiles(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics) {
	if r == nil || r.Config == nil || !slices.Contains(r.Config.AllowLossyTranslation, llmprotocol.LossyFieldUnreadableFile) {
		return request, nil
	}
	declared, annotated := r.declaredModelCapabilities(model)
	if !annotated || declared.Supports(llmprotocol.CapabilityFileInput) {
		return request, nil
	}
	return llmprotocol.ProjectUnreadableFiles(request, target)
}
