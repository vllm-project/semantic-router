package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// anthropicServerToolStream is a server tool block the decoder holds from its
// content_block_start to its content_block_stop, with the input its
// input_json_delta events streamed, so the block reaches the encoder whole.
type anthropicServerToolStream struct {
	index int
	block json.RawMessage
	input []byte
}

// decodeAnthropicServerToolEvent takes the events of a server tool block,
// reporting false for any other event.
func (decoder *anthropicStreamDecoder) decodeAnthropicServerToolEvent(
	wire anthropicEventWire,
	data []byte,
) ([]llmprotocol.Event, bool, error) {
	if wire.Type == "content_block_start" && wire.ContentBlock != nil && isAnthropicServerToolBlock(wire.ContentBlock.Type) {
		var event struct {
			ContentBlock json.RawMessage `json:"content_block"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, true, invalidProviderResponse("invalid_stream_content_block", "Anthropic server tool block is invalid")
		}
		decoder.serverTool = &anthropicServerToolStream{index: anthropicEventIndex(wire), block: event.ContentBlock}
		return nil, true, nil
	}
	held := decoder.serverTool
	if held == nil || anthropicEventIndex(wire) != held.index {
		return nil, false, nil
	}
	switch wire.Type {
	case "content_block_delta":
		if wire.Delta == nil || wire.Delta.Type != "input_json_delta" {
			return nil, true, invalidProviderResponse("invalid_stream_delta", "Anthropic server tool blocks stream only their input")
		}
		held.input = append(held.input, wire.Delta.PartialJSON...)
		return nil, true, nil
	case "content_block_stop":
		decoder.serverTool = nil
		return decoder.emitAnthropicServerTool(*held)
	}
	return nil, false, nil
}

func (decoder *anthropicStreamDecoder) emitAnthropicServerTool(held anthropicServerToolStream) ([]llmprotocol.Event, bool, error) {
	block := held.block
	if len(bytes.TrimSpace(held.input)) > 0 {
		var err error
		if block, err = withAnthropicServerToolInput(block, held.input); err != nil {
			return nil, true, err
		}
	}
	var identity struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(block, &identity)
	started, _, err := decoder.emitAnthropicEvent(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: held.index, ItemID: identity.ID, Role: llmprotocol.RoleAssistant,
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentServerTool, Raw: block},
	})
	if err != nil {
		return nil, true, err
	}
	completed, _, err := decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: held.index})
	return append(started, completed...), true, err
}

// withAnthropicServerToolInput sets a block's input to what its deltas
// streamed, and splitAnthropicServerToolInput undoes it for a stream.
func withAnthropicServerToolInput(block json.RawMessage, input []byte) (json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(block, &fields); err != nil || !json.Valid(input) {
		return nil, invalidProviderResponse("invalid_stream_tool_arguments", "Anthropic server tool input is not JSON")
	}
	fields["input"] = append(json.RawMessage(nil), input...)
	return json.Marshal(fields)
}

func splitAnthropicServerToolInput(block json.RawMessage) (json.RawMessage, string) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(block, &fields) != nil {
		return block, ""
	}
	input := bytes.TrimSpace(fields["input"])
	if len(input) == 0 || bytes.Equal(input, []byte("{}")) {
		return block, ""
	}
	fields["input"] = json.RawMessage(`{}`)
	start, err := json.Marshal(fields)
	if err != nil {
		return block, ""
	}
	return start, string(input)
}

// encodeAnthropicServerToolStart writes a server tool block as Anthropic
// streams one: its start with an empty input, then the input as one delta.
func encodeAnthropicServerToolStart(frames [][]byte, blockIndex int, content *llmprotocol.Content) ([][]byte, error) {
	if content == nil || content.Kind != llmprotocol.ContentServerTool {
		return frames, nil
	}
	_, input := splitAnthropicServerToolInput(content.Raw)
	if input == "" {
		return frames, nil
	}
	delta, _, err := encodeAnthropicWireFrame(anthropicEventWire{
		Type: "content_block_delta", Index: anthropicIndex(blockIndex),
		Delta: &anthropicDeltaWire{Type: "input_json_delta", PartialJSON: input},
	})
	return append(frames, delta...), err
}

// encodeAnthropicCitationDeltas writes each citation a text delta carries.
func encodeAnthropicCitationDeltas(frames [][]byte, blockIndex int, content *llmprotocol.Content) ([][]byte, error) {
	if content == nil || len(content.Raw) == 0 {
		return frames, nil
	}
	var citations []json.RawMessage
	if err := json.Unmarshal(content.Raw, &citations); err != nil {
		return frames, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_citations", "text citations are not a JSON array", err)
	}
	for _, citation := range citations {
		delta, _, err := encodeAnthropicWireFrame(anthropicEventWire{
			Type: "content_block_delta", Index: anthropicIndex(blockIndex),
			Delta: &anthropicDeltaWire{Type: "citations_delta", Citation: citation},
		})
		if err != nil {
			return frames, err
		}
		frames = append(frames, delta...)
	}
	return frames, nil
}
