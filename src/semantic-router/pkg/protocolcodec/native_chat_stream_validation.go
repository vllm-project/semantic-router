package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ValidateNativeChatStream validates a locally assembled Chat stream without
// flattening its alternatives. Every original frame first passes strict wire
// decoding; each choice then runs through the ordinary single-choice lifecycle
// decoder. This preserves Ratings' native multi-choice output without granting
// an alternate path around provider field validation or output limits.
func (engine *Engine) ValidateNativeChatStream(body []byte) (llmprotocol.Diagnostics, error) {
	if err := engine.ValidateEncodedResponse(body, true); err != nil {
		return nil, err
	}
	policy := engine.strictStreamPolicy()
	decoders := make(map[int]llmprotocol.StreamDecoder)
	var diagnostics llmprotocol.Diagnostics
	done := false
	first := true
	for len(body) > 0 {
		end, complete := completeSSEFrame(body)
		if !complete {
			end = len(body)
		}
		frame, err := parseSSEFrameAtPosition(body[:end], policy.Limits.SSEFrameBytes, first)
		if err != nil {
			return nil, err
		}
		first = false
		body = body[end:]
		if !frame.HasData {
			continue
		}
		if done {
			return nil, invalidProviderResponse("stream_event_after_terminal", "Chat stream emitted data after its terminal sentinel")
		}
		if bytes.Equal(bytes.TrimSpace(frame.Data), []byte("[DONE]")) {
			done = true
			for _, decoder := range decoders {
				_, warnings, decodeErr := decoder.Push([]byte("data: [DONE]\n\n"))
				diagnostics = appendDiagnostics(diagnostics, warnings, policy.Limits.Diagnostics)
				if decodeErr != nil {
					return nil, decodeErr
				}
			}
			continue
		}
		var chunk chatChunkWire
		if decodeErr := decodeProviderWire(frame.Data, &chunk, policy); decodeErr != nil {
			return nil, decodeErr
		}
		if envelopeErr := validateChatStreamEnvelope(chunk); envelopeErr != nil {
			return nil, envelopeErr
		}
		warnings, err := validateNativeChatChoices(chunk, decoders, policy)
		diagnostics = appendDiagnostics(diagnostics, warnings, policy.Limits.Diagnostics)
		if err != nil {
			return nil, err
		}
	}
	if !done || len(decoders) == 0 {
		return nil, invalidProviderResponse("incomplete_native_stream", "Chat stream did not complete any choices")
	}
	for _, decoder := range decoders {
		_, warnings, err := decoder.Finalize(nil)
		diagnostics = appendDiagnostics(diagnostics, warnings, policy.Limits.Diagnostics)
		if err != nil {
			return nil, err
		}
	}
	return diagnostics, nil
}

func validateNativeChatChoices(chunk chatChunkWire, decoders map[int]llmprotocol.StreamDecoder, policy llmprotocol.Policy) (llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	// A native stream is rendered only from successfully completed candidates.
	// An error envelope cannot be silently discarded, and usage cannot precede
	// the choice lifecycle it describes.
	if chunk.Error != nil {
		return nil, invalidProviderResponse("native_stream_error", "native Chat stream contains an error response")
	}
	if len(chunk.Choices) == 0 && len(decoders) == 0 {
		return nil, invalidProviderResponse("invalid_native_stream_order", "native Chat stream metadata precedes its choices")
	}
	if len(chunk.Choices) == 0 {
		encoded, err := json.Marshal(chunk)
		if err != nil {
			return nil, err
		}
		for _, decoder := range decoders {
			_, warnings, decodeErr := decoder.Push(append(append([]byte("data: "), encoded...), '\n', '\n'))
			diagnostics = appendDiagnostics(diagnostics, warnings, policy.Limits.Diagnostics)
			if decodeErr != nil {
				return nil, decodeErr
			}
		}
		return diagnostics, nil
	}
	seen := make(map[int]bool, len(chunk.Choices))
	for _, choice := range chunk.Choices {
		index := choice.Index
		if index < 0 || seen[index] || index >= policy.Limits.Candidates {
			return nil, invalidProviderResponse("invalid_native_choice_index", "native Chat choice index is invalid")
		}
		seen[index] = true
		decoder := decoders[index]
		if decoder == nil {
			decoder = (OpenAIChatCodec{}).NewDecoder(llmprotocol.StreamContext{Source: llmprotocol.OpenAIChatV1, Target: llmprotocol.OpenAIChatV1}, policy)
			decoders[index] = decoder
		}
		single := chunk
		choice.Index = 0
		single.Choices = []chatChunkChoiceWire{choice}
		encoded, err := json.Marshal(single)
		if err != nil {
			return nil, err
		}
		_, warnings, err := decoder.Push(append(append([]byte("data: "), encoded...), '\n', '\n'))
		diagnostics = appendDiagnostics(diagnostics, warnings, policy.Limits.Diagnostics)
		if err != nil {
			return nil, err
		}
	}
	return diagnostics, nil
}
