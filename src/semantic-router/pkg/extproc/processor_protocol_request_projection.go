package extproc

import (
	"fmt"
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
	projected, systemDiagnostics := r.projectSystemMessages(request, target)
	// An OpenAI-hosted Responses target leaves out unsigned thinking, so the
	// signatures go before that projection runs.
	projected, signatureDiagnostics, err := r.projectReasoningSignatures(projected, target)
	if err != nil {
		return request, nil, err
	}
	projected, reasoningDiagnostics := r.projectResponsesReasoningForBackend(projected, model, target, shadows...)
	projected, budgetDiagnostics, err := r.projectUnsupportedReasoningBudget(projected, model, target)
	if err != nil {
		return projected, nil, err
	}
	projected, verbosityDiagnostics := llmprotocol.ProjectTextVerbosity(projected, target)
	projected, toolChoiceDiagnostics := llmprotocol.ProjectToolChoiceNone(projected, target)
	projected, fileDiagnostics := r.projectUnreadableFiles(projected, model, target)
	projected, storeDiagnostics := r.projectUnstoredRequest(projected, target)
	projected, capabilityDiagnostics := r.projectUnsupportedCapabilities(projected, model, target)
	projected, otherDiagnostics, err := r.projectAnthropicRequestForBackendWithDiagnostics(projected, model, target)
	return projected, slices.Concat(systemDiagnostics, signatureDiagnostics, reasoningDiagnostics, budgetDiagnostics, verbosityDiagnostics,
		toolChoiceDiagnostics, fileDiagnostics, storeDiagnostics, capabilityDiagnostics, otherDiagnostics), err
}

// projectUnsupportedReasoningBudget follows the same projection path during
// candidate qualification and final dispatch. A non-vLLM Chat provider fails
// closed unless the operator explicitly accepts dropping its budget.
func (r *OpenAIRouter) projectUnsupportedReasoningBudget(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	if request.ReasoningBudgetTokens == nil || target != llmprotocol.OpenAIChatV1 {
		return request, nil, nil
	}
	providerType, err := r.providerTypeForModel(model)
	if err != nil {
		return request, nil, err
	}
	if providerType == vllmProviderType {
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
func (r *OpenAIRouter) providerTypeForModel(model string) (string, error) {
	if r == nil || r.Config == nil {
		return "", nil
	}
	_, endpointName, found, err := r.Config.ResolvePrimaryBackendForModel(model)
	if err != nil {
		return "", fmt.Errorf("resolve primary backend for model %q: %w", model, err)
	}
	if !found {
		return "", fmt.Errorf("model %q has no configured backend", model)
	}
	profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName)
	if profileErr != nil {
		return "", fmt.Errorf("resolve provider profile for model %q: %w", model, profileErr)
	}
	if profile != nil {
		if providerType, typeErr := profile.ProviderType(); typeErr == nil {
			return strings.ToLower(strings.TrimSpace(providerType)), nil
		}
		return strings.ToLower(strings.TrimSpace(profile.Type)), nil
	}
	if endpoint, exists := r.Config.GetEndpointByName(endpointName); exists {
		return strings.ToLower(strings.TrimSpace(endpoint.Type)), nil
	}
	return "", fmt.Errorf("provider endpoint %q for model %q was not found", endpointName, model)
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

// projectReasoningSignatures sends signed thinking in the history to a backend
// whose wire format cannot carry the signature: without the signature once the
// operator accepts that loss, and otherwise not at all, with a refusal that
// names the client's own field. An Anthropic backend keeps every signature.
func (r *OpenAIRouter) projectReasoningSignatures(
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	if available, ok := r.codecCapabilitiesForFormat(target); !ok || available.Supports(llmprotocol.CapabilityReasoningSignature) {
		return request, nil, nil
	}
	if r == nil || r.Config == nil || !slices.Contains(r.Config.AllowLossyTranslation, llmprotocol.LossyFieldReasoningSignature) {
		return request, nil, llmprotocol.RefuseReasoningSignatures(request)
	}
	projected, diagnostics := llmprotocol.ProjectReasoningSignatures(request, target)
	return projected, diagnostics, nil
}

// projectSystemMessages sends a Chat backend one system message, first, and
// the system and developer messages inside the conversation as user text, once
// the operator accepts that loss. It runs first, while the instructions'
// positions still describe the conversation, and before the developer-role
// projection, which then finds no developer message to report.
func (r *OpenAIRouter) projectSystemMessages(
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics) {
	if r == nil || r.Config == nil || !slices.Contains(r.Config.AllowLossyTranslation, llmprotocol.LossyFieldSystemMessageRole) {
		return request, nil
	}
	return llmprotocol.ProjectSystemMessages(request, target)
}

// projectUnsupportedCapabilities sends a model the request its
// unsupported_capabilities allow: without the sampling controls it rejects,
// such as min_p for a vLLM backend that runs speculative decoding, and with
// developer messages as system messages when it lacks the developer role, as
// DeepSeek V4 does. Selection, dispatch and every fallback candidate project
// through here, so another model still receives them.
func (r *OpenAIRouter) projectUnsupportedCapabilities(
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
	projected, samplingDiagnostics := llmprotocol.ProjectUnsupportedSampling(request, unsupported, target)
	projected, developerDiagnostics := llmprotocol.ProjectUnsupportedDeveloperRole(projected, unsupported, target)
	return projected, slices.Concat(samplingDiagnostics, developerDiagnostics)
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
