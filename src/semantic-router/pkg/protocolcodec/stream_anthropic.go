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
	return decoder.decodeEvent(wire, frame)
}

func (decoder *anthropicStreamDecoder) decodeEvent(
	wire anthropicEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	switch wire.Type {
	case "message_start":
		return decoder.emitAnthropicEvent(decodeAnthropicMessageStart(wire))
	case "content_block_start":
		event, decodeErr := decodeAnthropicContentStart(wire)
		if decodeErr != nil {
			return nil, nil, decodeErr
		}
		return decoder.emitAnthropicEvent(event)
	case "content_block_delta":
		event, decodeErr := decodeAnthropicContentDelta(wire)
		if decodeErr != nil {
			return nil, nil, decodeErr
		}
		return decoder.emitAnthropicEvent(event)
	case "content_block_stop":
		return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: wire.Index})
	case "message_delta":
		return decoder.decodeAnthropicMessageDelta(wire)
	case "message_stop":
		return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
	case "error":
		return decoder.emitAnthropicEvent(decodeAnthropicStreamError(wire))
	case "ping":
		return nil, nil, nil
	default:
		return decoder.decodeUnknownAnthropicEvent(frame)
	}
}

func decodeAnthropicMessageStart(wire anthropicEventWire) llmprotocol.Event {
	event := llmprotocol.Event{Type: llmprotocol.EventResponseStarted}
	if wire.Message == nil {
		return event
	}
	event.ResponseID, event.Model = wire.Message.ID, wire.Message.Model
	if wire.Message.Usage != nil {
		usage := decodeAnthropicStreamUsage(*wire.Message.Usage, true)
		event.Usage = &usage
	}
	return event
}

func (decoder *anthropicStreamDecoder) decodeAnthropicMessageDelta(
	wire anthropicEventWire,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if wire.Delta != nil && wire.Delta.StopReason != nil {
		decoder.stop = decodeAnthropicStop(*wire.Delta.StopReason)
	}
	if wire.Usage == nil {
		return nil, nil, nil
	}
	usage := decodeAnthropicStreamUsage(*wire.Usage, false)
	return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
}

func decodeAnthropicStreamError(wire anthropicEventWire) llmprotocol.Event {
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
	if wire.Error != nil {
		protocolError.Category = decodeProviderErrorCategory(wire.Error.Type)
		protocolError.Code, protocolError.Message = wire.Error.Type, wire.Error.Message
	}
	return llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed, Error: protocolError,
		StopReason: llmprotocol.StopError, Failure: llmprotocol.FailureTransport,
	}
}

func (decoder *anthropicStreamDecoder) decodeUnknownAnthropicEvent(
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if decoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || decoder.context.Source != decoder.context.Target {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Anthropic stream event is unsupported", nil)
	}
	return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque, Opaque: append([]byte(nil), frame...)})
}

func (decoder *anthropicStreamDecoder) emitAnthropicEvent(
	event llmprotocol.Event,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	normalized, err := decoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.Event{normalized}, nil, nil
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
		reasoning := int64(0)
		if wire.OutputTokensDetails != nil {
			reasoning = wire.OutputTokensDetails.ThinkingTokens
		}
		other := wire.OutputTokens - reasoning
		if other < 0 {
			other = 0
		}
		usage.OutputReasoning = authoritative(reasoning)
		usage.OutputOther = authoritative(other)
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
	if event.Type == llmprotocol.EventResponseCompleted {
		return encoder.encodeAnthropicCompletion(event)
	}
	if event.Type == llmprotocol.EventProviderOpaque {
		return encoder.encodeAnthropicOpaque(event)
	}
	if isAnthropicContentEvent(event.Type) {
		return encoder.encodeAnthropicContentEvent(event)
	}
	return encoder.encodeAnthropicLifecycleEvent(event)
}

func isAnthropicContentEvent(eventType llmprotocol.EventType) bool {
	return eventType == llmprotocol.EventOutputItemStarted ||
		eventType == llmprotocol.EventOutputTextDelta ||
		eventType == llmprotocol.EventReasoningDelta ||
		eventType == llmprotocol.EventToolCallDelta
}

func (encoder *anthropicStreamEncoder) encodeAnthropicContentEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.Type == llmprotocol.EventOutputTextDelta {
		return encoder.encodeAnthropicTextDelta(event)
	}
	wire := anthropicEventWire{}
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		wire = encoder.encodeAnthropicItemStart(event)
	case llmprotocol.EventReasoningDelta:
		wire = encodeAnthropicReasoningDelta(event)
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		wire.Type, wire.Index, wire.Delta = "content_block_delta", event.ItemIndex, &anthropicDeltaWire{Type: "input_json_delta", PartialJSON: event.ToolCall.Arguments}
	default:
		return nil, nil, nil
	}
	return encodeAnthropicWireFrame(wire)
}

func (encoder *anthropicStreamEncoder) encodeAnthropicLifecycleEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var wire anthropicEventWire
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire = encodeAnthropicMessageStart(event)
	case llmprotocol.EventOutputItemCompleted:
		wire.Type, wire.Index = "content_block_stop", event.ItemIndex
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		wire.Type, wire.Usage = "message_delta", encodeAnthropicUsage(*event.Usage)
	case llmprotocol.EventResponseFailed:
		failed, err := encoder.encodeAnthropicFailure(event)
		if err != nil {
			return nil, nil, err
		}
		wire = failed
	default:
		return nil, nil, nil
	}
	return encodeAnthropicWireFrame(wire)
}

func encodeAnthropicWireFrame(wire anthropicEventWire) ([][]byte, llmprotocol.Diagnostics, error) {
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func encodeAnthropicMessageStart(event llmprotocol.Event) anthropicEventWire {
	usage := &anthropicUsageWire{}
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		usage = encodeAnthropicUsage(*event.Usage)
	}
	return anthropicEventWire{
		Type: "message_start",
		Message: &anthropicResponseWire{
			ID: event.ResponseID, Type: "message", Role: "assistant", Model: event.Model,
			Content: json.RawMessage(`[]`), Usage: usage,
		},
	}
}

func (encoder *anthropicStreamEncoder) encodeAnthropicItemStart(event llmprotocol.Event) anthropicEventWire {
	block := &anthropicContentWire{Type: "text", Text: ""}
	kind := llmprotocol.ContentText
	if event.ToolCall != nil {
		block.Type, block.ID, block.Name, block.Input = "tool_use", event.ToolCall.ID, event.ToolCall.Name, json.RawMessage(`{}`)
		kind = llmprotocol.ContentToolCall
	} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		block.Type, block.Text, block.Thinking, block.Signature = "thinking", "", "", event.Content.Signature
		kind = llmprotocol.ContentReasoning
	}
	encoder.blocks[event.ItemIndex] = kind
	return anthropicEventWire{Type: "content_block_start", Index: event.ItemIndex, ContentBlock: block}
}

func encodeAnthropicReasoningDelta(event llmprotocol.Event) anthropicEventWire {
	delta := &anthropicDeltaWire{Type: "thinking_delta", Thinking: event.Delta}
	if event.Content != nil && event.Content.Signature != "" {
		delta = &anthropicDeltaWire{Type: "signature_delta", Signature: event.Content.Signature}
	}
	return anthropicEventWire{Type: "content_block_delta", Index: event.ItemIndex, Delta: delta}
}

func (encoder *anthropicStreamEncoder) encodeAnthropicCompletion(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.Usage == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
	}
	stop := encodeAnthropicStop(event.StopReason)
	delta := anthropicEventWire{Type: "message_delta", Delta: &anthropicDeltaWire{Type: "message_delta", StopReason: &stop}, Usage: encodeAnthropicUsage(*event.Usage)}
	first, err := encodeSSE(delta.Type, delta)
	if err != nil {
		return nil, nil, err
	}
	encoder.terminal = true
	stopEvent := anthropicEventWire{Type: "message_stop"}
	second, err := encodeSSE(stopEvent.Type, stopEvent)
	return [][]byte{first, second}, nil, err
}

func (encoder *anthropicStreamEncoder) encodeAnthropicFailure(event llmprotocol.Event) (anthropicEventWire, error) {
	if event.Error == nil {
		return anthropicEventWire{}, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	return anthropicEventWire{
		Type:  "error",
		Error: &anthropicErrorWire{Type: canonicalAnthropicErrorType(event.Error), Message: event.Error.Message},
	}, nil
}

func (encoder *anthropicStreamEncoder) encodeAnthropicOpaque(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
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
	protocolError := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"stream_incomplete",
		"stream ended before completion",
		reason,
	)
	wire := anthropicEventWire{Type: "error", Error: &anthropicErrorWire{
		Type: canonicalAnthropicErrorType(protocolError), Message: protocolError.Message,
	}}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}
