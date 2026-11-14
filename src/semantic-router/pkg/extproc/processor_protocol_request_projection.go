package extproc

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// projectRequestForBackend applies portable request projections before model
// qualification and again before encoding. Only the encoding pass publishes
// diagnostics, so failed candidate probes cannot leak warnings to the client.
func (r *OpenAIRouter) projectRequestForBackend(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, error) {
	projected, _, err := r.projectRequestForBackendWithDiagnostics(request, model, target)
	return projected, err
}

func (r *OpenAIRouter) projectRequestForBackendWithDiagnostics(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	projected, verbosityDiagnostics := llmprotocol.ProjectTextVerbosity(request, target)
	projected, fileDiagnostics := r.projectUnreadableFiles(projected, model, target)
	projected, storeDiagnostics := r.projectUnstoredRequest(projected, target)
	projected, samplingDiagnostics := r.projectUnsupportedSampling(projected, model, target)
	projected, otherDiagnostics, err := r.projectAnthropicRequestForBackendWithDiagnostics(projected, model, target)
	return projected, slices.Concat(verbosityDiagnostics, fileDiagnostics, storeDiagnostics, samplingDiagnostics, otherDiagnostics), err
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
