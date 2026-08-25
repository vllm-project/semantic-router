package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type anthropicStreamDecoder struct {
	streamState
	framer sseFramer
}
type anthropicStreamEncoder struct {
	streamState
	blocks map[int]llmprotocol.ContentKind
}

func (AnthropicMessagesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &anthropicStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}
}

func (AnthropicMessagesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &anthropicStreamEncoder{streamState: streamState{context: context, policy: policy}, blocks: make(map[int]llmprotocol.ContentKind)}
}

type anthropicEventWire struct {
	Type         string                 `json:"type"`
	Message      *anthropicResponseWire `json:"message,omitempty"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *anthropicContentWire  `json:"content_block,omitempty"`
	Delta        *anthropicDeltaWire    `json:"delta,omitempty"`
	Usage        *anthropicUsageWire    `json:"usage,omitempty"`
	Error        *anthropicErrorWire    `json:"error,omitempty"`
}

type anthropicDeltaWire struct {
	Type         string  `json:"type"`
	Text         string  `json:"text,omitempty"`
	Thinking     string  `json:"thinking,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
	Signature    string  `json:"signature,omitempty"`
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

func (decoder *anthropicStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *anthropicStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	var wire anthropicEventWire
	if err := decodeProviderWire(parsed.Data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = parsed.Event
	}
	events := make([]llmprotocol.Event, 0, 2)
	switch wire.Type {
	case "message_start":
		event := llmprotocol.Event{Type: llmprotocol.EventResponseStarted}
		if wire.Message != nil {
			event.ResponseID, event.Model = wire.Message.ID, wire.Message.Model
			if wire.Message.Usage != nil {
				usage := decodeAnthropicStreamUsage(*wire.Message.Usage, true)
				event.Usage = &usage
			}
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_start":
		event, decodeErr := decodeAnthropicContentStart(wire)
		if decodeErr != nil {
			return nil, nil, decodeErr
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_delta":
		event, decodeErr := decodeAnthropicContentDelta(wire)
		if decodeErr != nil {
			return nil, nil, decodeErr
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_stop":
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: wire.Index})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "message_delta":
		if wire.Delta != nil && wire.Delta.StopReason != nil {
			decoder.stop = decodeAnthropicStop(*wire.Delta.StopReason)
		}
		if wire.Usage != nil {
			usage := decodeAnthropicStreamUsage(*wire.Usage, false)
			normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, normalized)
		}
	case "message_stop":
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "error":
		protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
		if wire.Error != nil {
			protocolError.Code, protocolError.Message = wire.Error.Type, wire.Error.Message
		}
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "ping":
		return nil, nil, nil
	default:
		if decoder.policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat && decoder.context.Source == decoder.context.Target {
			normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque, Opaque: append([]byte(nil), frame...)})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, normalized)
		} else {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Anthropic stream event is unsupported", nil)
		}
	}
	return events, nil, nil
}

func decodeAnthropicContentStart(wire anthropicEventWire) (llmprotocol.Event, error) {
	event := llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: wire.Index, Role: llmprotocol.RoleAssistant}
	if wire.ContentBlock == nil {
		return event, nil
	}
	event.ItemID = wire.ContentBlock.ID
	switch wire.ContentBlock.Type {
	case "tool_use":
		event.ToolCall = &llmprotocol.ToolCall{ID: wire.ContentBlock.ID, Name: wire.ContentBlock.Name, Arguments: string(wire.ContentBlock.Input)}
	case "thinking":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Signature: wire.ContentBlock.Signature}
	case "text":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText}
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic stream content block is unsupported", nil)
	}
	return event, nil
}

func decodeAnthropicContentDelta(wire anthropicEventWire) (llmprotocol.Event, error) {
	if wire.Delta == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_delta", "Anthropic stream delta is missing", nil)
	}
	event := llmprotocol.Event{ItemIndex: wire.Index}
	switch wire.Delta.Type {
	case "text_delta":
		event.Type, event.Delta, event.Content = llmprotocol.EventOutputTextDelta, wire.Delta.Text, &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta.Text}
	case "thinking_delta":
		event.Type, event.Delta, event.Content = llmprotocol.EventReasoningDelta, wire.Delta.Thinking, &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: wire.Delta.Thinking}
	case "input_json_delta":
		event.Type, event.ToolCall = llmprotocol.EventToolCallDelta, &llmprotocol.ToolCall{Arguments: wire.Delta.PartialJSON}
	case "signature_delta":
		event.Type, event.Content = llmprotocol.EventReasoningDelta, &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Signature: wire.Delta.Signature}
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_delta", "Anthropic stream delta is unsupported", nil)
	}
	return event, nil
}

func decodeAnthropicStreamUsage(wire anthropicUsageWire, initial bool) llmprotocol.Usage {
	usage := llmprotocol.Usage{State: llmprotocol.UsageAvailable}
	if initial || wire.InputTokens > 0 || wire.CacheReadInputTokens > 0 || wire.CacheCreationInputTokens > 0 {
		uncached := wire.InputTokens - wire.CacheReadInputTokens - wire.CacheCreationInputTokens
		if uncached < 0 {
			uncached = 0
		}
		usage.InputUncached = authoritative(uncached)
		usage.InputCacheRead = authoritative(wire.CacheReadInputTokens)
		usage.InputCacheWrite = authoritative(wire.CacheCreationInputTokens)
		usage.InputTotal = authoritative(wire.InputTokens)
	}
	if wire.OutputTokens > 0 || !initial {
		usage.OutputOther = authoritative(wire.OutputTokens)
		usage.OutputTotal = authoritative(wire.OutputTokens)
	}
	return usage
}

func (decoder *anthropicStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *anthropicStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, pushErr := encoder.next(event)
	if pushErr != nil {
		return nil, nil, pushErr
	}
	event = normalized
	wire := anthropicEventWire{}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire.Type = "message_start"
		usage := &anthropicUsageWire{}
		if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
			usage = encodeAnthropicUsage(*event.Usage)
		}
		wire.Message = &anthropicResponseWire{ID: event.ResponseID, Type: "message", Role: "assistant", Model: event.Model, Content: json.RawMessage(`[]`), Usage: usage}
	case llmprotocol.EventOutputItemStarted:
		wire.Type, wire.Index = "content_block_start", event.ItemIndex
		block := &anthropicContentWire{Type: "text", Text: ""}
		if event.ToolCall != nil {
			block.Type, block.ID, block.Name, block.Input = "tool_use", event.ToolCall.ID, event.ToolCall.Name, json.RawMessage(`{}`)
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentToolCall
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			block.Type, block.Text, block.Thinking, block.Signature = "thinking", "", "", event.Content.Signature
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentReasoning
		} else {
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentText
		}
		wire.ContentBlock = block
	case llmprotocol.EventOutputTextDelta:
		return encoder.encodeAnthropicTextDelta(event)
	case llmprotocol.EventReasoningDelta:
		wire.Type, wire.Index = "content_block_delta", event.ItemIndex
		if event.Content != nil && event.Content.Signature != "" {
			wire.Delta = &anthropicDeltaWire{Type: "signature_delta", Signature: event.Content.Signature}
		} else {
			wire.Delta = &anthropicDeltaWire{Type: "thinking_delta", Thinking: event.Delta}
		}
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		wire.Type, wire.Index, wire.Delta = "content_block_delta", event.ItemIndex, &anthropicDeltaWire{Type: "input_json_delta", PartialJSON: event.ToolCall.Arguments}
	case llmprotocol.EventOutputItemCompleted:
		wire.Type, wire.Index = "content_block_stop", event.ItemIndex
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		wire.Type, wire.Usage = "message_delta", encodeAnthropicUsage(*event.Usage)
	case llmprotocol.EventResponseCompleted:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
		}
		stop := encodeAnthropicStop(event.StopReason)
		delta := anthropicEventWire{Type: "message_delta", Delta: &anthropicDeltaWire{Type: "message_delta", StopReason: &stop}, Usage: encodeAnthropicUsage(*event.Usage)}
		first, err := encodeSSE(delta.Type, delta)
		if err != nil {
			return nil, nil, err
		}
		wire.Type = "message_stop"
		encoder.terminal = true
		second, err := encodeSSE(wire.Type, wire)
		return [][]byte{first, second}, nil, err
	case llmprotocol.EventResponseFailed:
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		wire.Type = "error"
		wire.Error = &anthropicErrorWire{Type: event.Error.Code, Message: event.Error.Message}
		encoder.terminal = true
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		return nil, nil, nil
	}
	frame, pushErr := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, pushErr
}

func (encoder *anthropicStreamEncoder) encodeAnthropicTextDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if event.Content != nil && len(event.Content.Citations) > 0 {
		if err := appendLossy(
			&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
			"content.citations", "Messages cannot represent URL citations",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		if err := appendLossy(
			&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
			"content.refusal", "Messages represents refusal as ordinary text",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := anthropicEventWire{
		Type: "content_block_delta", Index: event.ItemIndex,
		Delta: &anthropicDeltaWire{Type: "text_delta", Text: event.Delta},
	}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, diagnostics, err
}

func (encoder *anthropicStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	wire := anthropicEventWire{Type: "error", Error: &anthropicErrorWire{Type: "stream_incomplete", Message: "stream ended before completion"}}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}
