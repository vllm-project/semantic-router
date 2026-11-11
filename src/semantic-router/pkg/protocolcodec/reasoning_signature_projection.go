package protocolcodec

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// projectReasoningSignature drops reasoning signatures from a buffered response
// whose client format cannot carry them, but only when the operator accepted
// that loss. Without the opt-in the capability check that follows still
// refuses the response. The reasoning text is kept.
func projectReasoningSignature(
	policy llmprotocol.Policy,
	source, target llmprotocol.WireFormat,
	capabilities llmprotocol.CapabilitySet,
	response *llmprotocol.Response,
) llmprotocol.Diagnostics {
	if capabilities.Supports(llmprotocol.CapabilityReasoningSignature) ||
		!slices.Contains(policy.AllowedLossyFields, llmprotocol.LossyFieldReasoningSignature) {
		return nil
	}
	output, dropped := withoutReasoningSignatures(response.Output)
	var alternatives [][]llmprotocol.OutputItem
	for index, items := range response.Alternatives {
		projected, droppedAlternative := withoutReasoningSignatures(items)
		if !droppedAlternative {
			continue
		}
		if alternatives == nil {
			alternatives = append([][]llmprotocol.OutputItem(nil), response.Alternatives...)
		}
		alternatives[index], dropped = projected, true
	}
	if !dropped {
		return nil
	}
	response.Output = output
	if alternatives != nil {
		response.Alternatives = alternatives
	}
	var diagnostics llmprotocol.Diagnostics
	_ = appendLossy(&diagnostics, policy, source, target, llmprotocol.LossyFieldReasoningSignature,
		"the client format cannot carry a reasoning signature; the reasoning text is kept")
	return diagnostics
}

// withoutReasoningSignatures never edits the caller's slices.
func withoutReasoningSignatures(items []llmprotocol.OutputItem) ([]llmprotocol.OutputItem, bool) {
	dropped := false
	projected := make([]llmprotocol.OutputItem, len(items))
	for index, item := range items {
		projected[index] = item
		copied := false
		for blockIndex, block := range item.Content {
			if block.Kind != llmprotocol.ContentReasoning || block.Signature == "" {
				continue
			}
			if !copied {
				projected[index].Content = append([]llmprotocol.Content(nil), item.Content...)
				copied = true
			}
			projected[index].Content[blockIndex].Signature = ""
			dropped = true
		}
	}
	return projected, dropped
}
