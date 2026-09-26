package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

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
	projected, otherDiagnostics, err := r.projectAnthropicRequestForBackendWithDiagnostics(projected, model, target)
	return projected, append(verbosityDiagnostics, otherDiagnostics...), err
}
