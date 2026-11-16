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

// projectReasoningContinuity drops reply blocks the client cannot represent.
// Other request continuity remains fail-closed; Responses blobs are dropped
// by request encoders for formats that cannot carry them.
func projectReasoningContinuity(
	source, target llmprotocol.WireFormat,
	capabilities llmprotocol.CapabilitySet,
	response *llmprotocol.Response,
) llmprotocol.Diagnostics {
	output, dropped := withoutReasoningDetails(response.Output, target, capabilities)
	var alternatives [][]llmprotocol.OutputItem
	for index, items := range response.Alternatives {
		projected, droppedAlternative := withoutReasoningDetails(items, target, capabilities)
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
func projectReasoningContinuityEvent(target llmprotocol.WireFormat, capabilities llmprotocol.CapabilitySet, event llmprotocol.Event) error {
	if event.Content == nil || reasoningDetailRepresentable(event.Content.ReasoningDetail, target, capabilities) {
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
func withoutReasoningDetails(items []llmprotocol.OutputItem, target llmprotocol.WireFormat, capabilities llmprotocol.CapabilitySet) ([]llmprotocol.OutputItem, bool) {
	dropped := false
	kept := make([]llmprotocol.OutputItem, 0, len(items))
	for _, item := range items {
		content := make([]llmprotocol.Content, 0, len(item.Content))
		for _, block := range item.Content {
			if reasoningDetailRepresentable(block.ReasoningDetail, target, capabilities) {
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

func reasoningDetailRepresentable(detail *llmprotocol.ReasoningDetail, target llmprotocol.WireFormat, capabilities llmprotocol.CapabilitySet) bool {
	if detail == nil {
		return true
	}
	if llmprotocol.IsResponsesReasoning(detail) {
		return target == llmprotocol.OpenAIResponsesV1
	}
	return capabilities.Supports(llmprotocol.CapabilityReasoningContinuity)
}

// requireReasoningRepresentation guards direct codec calls after the capability
// check: continuity wire representations are not interchangeable.
func requireReasoningRepresentation(target llmprotocol.WireFormat, contents []llmprotocol.Content) error {
	capabilities := llmprotocol.Capabilities()
	if target == llmprotocol.OpenAIChatV1 {
		capabilities = (OpenAIChatCodec{}).Capabilities()
	}
	for _, content := range contents {
		if !reasoningDetailRepresentable(content.ReasoningDetail, target, capabilities) {
			return rejectReasoningContinuity(target, llmprotocol.Capabilities(llmprotocol.CapabilityReasoningContinuity))
		}
	}
	return nil
}
