package protocolcodec

import (
	"bytes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func (decoder *anthropicStreamDecoder) anthropicMessageDeltaDiagnostics(wire anthropicEventWire) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	// The final message_delta reports applied context edits in a top-level
	// context_management member, a sibling of delta and usage rather than a
	// delta field. It only appears when the request carried the directive.
	if len(wire.ContextManagement) > 0 && !bytes.Equal(bytes.TrimSpace(wire.ContextManagement), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.context_management", "applied context edits describe upstream prompt trimming and have no neutral representation",
		)
	}
	if wire.Delta == nil {
		return diagnostics
	}
	if len(wire.Delta.Container) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Delta.Container), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.container", "container metadata has no protocol-neutral representation",
		)
	}
	if len(wire.Delta.StopDetails) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Delta.StopDetails), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.stop_details", "refusal details have no protocol-neutral representation",
		)
	}
	return diagnostics
}
