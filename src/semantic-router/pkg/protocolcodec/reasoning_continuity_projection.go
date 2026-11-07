package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// rejectReasoningContinuity guards codecs that cannot carry continuity blocks
// when they are called directly; the engine projects replies before them.
func rejectReasoningContinuity(format llmprotocol.WireFormat, required llmprotocol.CapabilitySet) error {
	if !required.Supports(llmprotocol.CapabilityReasoningContinuity) {
		return nil
	}
	return llmprotocol.RequireCapabilities(format, llmprotocol.Capabilities(), llmprotocol.Capabilities(llmprotocol.CapabilityReasoningContinuity))
}

// projectReasoningContinuity drops typed reasoning continuity blocks that the
// client format cannot carry, instead of failing a reply the provider has
// already produced. The plain reasoning text stays and the drop is diagnosed,
// so the loss is never silent. Requests do not project: a backend that cannot
// receive the blocks would lose the provider's tool-loop state, so request
// translation still fails its capability check.
func projectReasoningContinuity(
	source, target llmprotocol.WireFormat,
	capabilities llmprotocol.CapabilitySet,
	response *llmprotocol.Response,
) llmprotocol.Diagnostics {
	if capabilities.Supports(llmprotocol.CapabilityReasoningContinuity) {
		return nil
	}
	output, dropped := withoutReasoningDetails(response.Output)
	var alternatives [][]llmprotocol.OutputItem
	for index, items := range response.Alternatives {
		projected, droppedAlternative := withoutReasoningDetails(items)
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
	return reasoningContinuityDropped(source, target)
}

// projectReasoningContinuityEvent is the streaming counterpart. The returned
// error omits a delta that carries only a continuity block; the caller reports
// the drop once per stream.
func projectReasoningContinuityEvent(capabilities llmprotocol.CapabilitySet, event llmprotocol.Event) error {
	if capabilities.Supports(llmprotocol.CapabilityReasoningContinuity) ||
		event.Content == nil || event.Content.ReasoningDetail == nil {
		return nil
	}
	return ErrOmitStreamEvent
}

func reasoningContinuityDropped(source, target llmprotocol.WireFormat) llmprotocol.Diagnostics {
	return llmprotocol.Diagnostics{{
		Source: source, Target: target,
		Field:  "reasoning_details",
		Action: llmprotocol.DiagnosticDropped,
		Reason: "typed reasoning continuity blocks have no representation in the client format; the reasoning text is kept",
	}}
}

// withoutReasoningDetails never edits the caller's slices. An item left with
// no content is removed rather than rendered empty.
func withoutReasoningDetails(items []llmprotocol.OutputItem) ([]llmprotocol.OutputItem, bool) {
	dropped := false
	kept := make([]llmprotocol.OutputItem, 0, len(items))
	for _, item := range items {
		content := make([]llmprotocol.Content, 0, len(item.Content))
		for _, block := range item.Content {
			if block.ReasoningDetail == nil {
				content = append(content, block)
			}
		}
		if len(content) == len(item.Content) {
			kept = append(kept, item)
			continue
		}
		dropped = true
		if len(content) == 0 {
			continue
		}
		item.Content = content
		kept = append(kept, item)
	}
	if !dropped {
		return items, false
	}
	return kept, true
}
