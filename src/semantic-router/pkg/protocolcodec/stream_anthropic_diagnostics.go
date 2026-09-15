package protocolcodec

import (
	"bytes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Takes the whole frame because context_management sits on the event, as a
// sibling of delta rather than a member of it.
func (decoder *anthropicStreamDecoder) anthropicMessageDeltaDiagnostics(wire anthropicEventWire) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	if len(wire.ContextManagement) > 0 && !bytes.Equal(bytes.TrimSpace(wire.ContextManagement), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.context_management", "context editing echo is request metadata, not model output",
		)
	}
	delta := wire.Delta
	if delta == nil {
		return diagnostics
	}
	if len(delta.Container) > 0 && !bytes.Equal(bytes.TrimSpace(delta.Container), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.container", "container metadata has no protocol-neutral representation",
		)
	}
	if len(delta.StopDetails) > 0 && !bytes.Equal(bytes.TrimSpace(delta.StopDetails), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.stop_details", "refusal details have no protocol-neutral representation",
		)
	}
	return diagnostics
}
