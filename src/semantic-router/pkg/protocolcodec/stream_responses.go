package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesStreamDecoder struct {
	streamState
	framer                sseFramer
	nextAnnotationIndexes map[int]int
}
type responsesStreamEncoder struct {
	streamState
	startedItems     map[int]bool
	contentStarted   map[int]bool
	contentKinds     map[int]llmprotocol.ContentKind
	contentText      map[int]*strings.Builder
	contentCitations map[int][]llmprotocol.Citation
	wireSequence     uint64
}

func (OpenAIResponsesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &responsesStreamDecoder{
		streamState:           streamState{context: context, policy: policy},
		framer:                newSSEFramer(policy.Limits.SSEFrameBytes),
		nextAnnotationIndexes: make(map[int]int),
	}
}

func (OpenAIResponsesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &responsesStreamEncoder{
		streamState:      streamState{context: context, policy: policy},
		startedItems:     make(map[int]bool),
		contentStarted:   make(map[int]bool),
		contentKinds:     make(map[int]llmprotocol.ContentKind),
		contentText:      make(map[int]*strings.Builder),
		contentCitations: make(map[int][]llmprotocol.Citation),
	}
}

type responsesEventWire struct {
	Type            string                   `json:"type"`
	Sequence        uint64                   `json:"sequence_number,omitempty"`
	Response        *responsesResponseWire   `json:"response,omitempty"`
	Item            *responsesItemWire       `json:"item,omitempty"`
	ItemID          string                   `json:"item_id,omitempty"`
	OutputIndex     int                      `json:"output_index,omitempty"`
	ContentIndex    *int                     `json:"content_index,omitempty"`
	AnnotationIndex *int                     `json:"annotation_index,omitempty"`
	Delta           string                   `json:"delta,omitempty"`
	Text            string                   `json:"text,omitempty"`
	Part            *responsesContentWire    `json:"part,omitempty"`
	Annotation      *responsesAnnotationWire `json:"annotation,omitempty"`
	Name            string                   `json:"name,omitempty"`
	Error           *responsesErrorWire      `json:"error,omitempty"`
}

func (decoder *responsesStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
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

func (decoder *responsesStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		// A Responses terminal event is authoritative. Some compatible
		// transports append this sentinel for framing compatibility; it is not
		// a second semantic completion.
		if decoder.terminal {
			return nil, nil, nil
		}
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		return []llmprotocol.Event{event}, nil, nextErr
	}
	var wire responsesEventWire
	if err := decodeProviderWire(parsed.Data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = parsed.Event
	}
	event := llmprotocol.Event{ResponseID: decoder.context.ResponseID, Model: decoder.context.PublicModel, ItemIndex: wire.OutputIndex, ItemID: wire.ItemID, Delta: wire.Delta}
	switch wire.Type {
	case "response.created", "response.in_progress":
		if decoder.started {
			return nil, nil, nil
		}
		event.Type = llmprotocol.EventResponseStarted
		if wire.Response != nil {
			event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
		}
	case "response.output_item.added":
		event.Type = llmprotocol.EventOutputItemStarted
		if wire.Item != nil {
			event.ItemID = wire.Item.ID
			event.Role = llmprotocol.RoleAssistant
			if wire.Item.Type == "function_call" {
				event.ToolCall = &llmprotocol.ToolCall{ID: wire.Item.CallID, Name: wire.Item.Name, Arguments: wire.Item.Arguments}
			}
		}
	case "response.output_text.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta}
	case "response.output_text.annotation.added":
		if err := decoder.applyResponseAnnotation(&event, wire); err != nil {
			return nil, nil, err
		}
	case "response.refusal.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: wire.Delta}
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		event.Type = llmprotocol.EventReasoningDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: wire.Delta}
	case "response.function_call_arguments.delta":
		event.Type = llmprotocol.EventToolCallDelta
		call := decoder.toolCalls[wire.OutputIndex]
		if wire.Item != nil && wire.Item.CallID != "" {
			call.ID = wire.Item.CallID
		}
		if wire.Name != "" {
			call.Name = wire.Name
		}
		call.Arguments = wire.Delta
		event.ToolCall = &call
	case "response.output_item.done":
		if err := decoder.applyCompletedResponseItem(&event, wire); err != nil {
			return nil, nil, err
		}
	case "response.completed", "response.incomplete":
		event.Type = llmprotocol.EventResponseCompleted
		event.StopReason = llmprotocol.StopEndTurn
		if wire.Type == "response.incomplete" {
			event.StopReason = llmprotocol.StopMaxTokens
		}
		if wire.Response != nil && wire.Response.Usage != nil {
			usage := decodeResponsesUsage(*wire.Response.Usage)
			event.Usage = &usage
		}
	case "response.failed", "error":
		applyResponseFailure(&event, wire)
	case "response.content_part.added", "response.content_part.done", "response.output_text.done", "response.refusal.done", "response.function_call_arguments.done":
		return nil, nil, nil
	default:
		if decoder.policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat && decoder.context.Source == decoder.context.Target {
			event.Type, event.Opaque = llmprotocol.EventProviderOpaque, append([]byte(nil), frame...)
		} else {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Responses stream event is unsupported", nil)
		}
	}
	normalized, nextErr := decoder.next(event)
	return []llmprotocol.Event{normalized}, nil, nextErr
}

func (decoder *responsesStreamDecoder) applyResponseAnnotation(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	if wire.Annotation == nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_missing", "Responses citation event is missing its annotation", nil)
	}
	itemID, itemFound := decoder.itemIDs[wire.OutputIndex]
	if !itemFound || wire.ItemID == "" || wire.ItemID != itemID {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_item", "Responses citation event does not match its active output item", nil)
	}
	if wire.ContentIndex == nil || *wire.ContentIndex != 0 {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_content_index", "Responses citation event has an unsupported content index", nil)
	}
	expected := decoder.nextAnnotationIndexes[wire.OutputIndex]
	if wire.AnnotationIndex == nil || *wire.AnnotationIndex != expected {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_index", "Responses citation indexes must be monotonic and contiguous", nil)
	}
	citations, err := decodeResponsesAnnotations([]responsesAnnotationWire{*wire.Annotation})
	if err != nil {
		return err
	}
	event.Type = llmprotocol.EventOutputTextDelta
	event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
	decoder.nextAnnotationIndexes[wire.OutputIndex] = expected + 1
	return nil
}

func (decoder *responsesStreamDecoder) applyCompletedResponseItem(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	event.Type = llmprotocol.EventOutputItemCompleted
	if wire.Item == nil {
		return nil
	}
	event.ItemID = wire.Item.ID
	switch wire.Item.Type {
	case "function_call":
		event.ToolCall = &llmprotocol.ToolCall{ID: wire.Item.CallID, Name: wire.Item.Name, Arguments: wire.Item.Arguments}
	case "message":
		if decoder.itemKinds[wire.OutputIndex] == llmprotocol.ContentToolCall {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_kind_mismatch", "upstream completed a tool item as a message", nil)
		}
	case "reasoning":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses completed an unsupported output item", nil)
	}
	return nil
}

func applyResponseFailure(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseFailed
	event.StopReason = llmprotocol.StopError
	event.Error = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
	upstreamError := wire.Error
	if upstreamError == nil && wire.Response != nil {
		upstreamError = wire.Response.Error
	}
	if upstreamError != nil {
		event.Error.Code, event.Error.Message = upstreamError.Code, upstreamError.Message
	}
}

func (decoder *responsesStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *responsesStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	wire := responsesEventWire{ItemID: event.ItemID, OutputIndex: event.ItemIndex, Delta: event.Delta}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire.Type = "response.created"
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "in_progress"}
	case llmprotocol.EventOutputItemStarted:
		wire.Type = "response.output_item.added"
		wire.Item = &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "in_progress"}
		if event.ToolCall != nil {
			wire.Item.Type, wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = "function_call", event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
			wire.Item.Role = ""
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			wire.Item.Type, wire.Item.Role = "reasoning", ""
		} else if event.Content != nil {
			encoder.contentKinds[event.ItemIndex] = event.Content.Kind
		}
	case llmprotocol.EventOutputTextDelta:
		return encoder.encodeResponsesTextDelta(event)
	case llmprotocol.EventReasoningDelta:
		if event.Content != nil && event.Content.Signature != "" {
			var diagnostics llmprotocol.Diagnostics
			if lossyErr := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Responses cannot represent a signed reasoning delta"); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
			wire.Type = "response.reasoning_text.delta"
			wire.Sequence = encoder.nextWireSequence()
			frame, encodeErr := encodeSSE(wire.Type, wire)
			return [][]byte{frame}, diagnostics, encodeErr
		}
		wire.Type = "response.reasoning_text.delta"
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		wire.Type, wire.ItemID, wire.Name, wire.Delta = "response.function_call_arguments.delta", event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
	case llmprotocol.EventOutputItemCompleted:
		return encoder.encodeCompletedResponsesItem(event)
	case llmprotocol.EventUsageUpdated:
		return nil, nil, nil
	case llmprotocol.EventResponseCompleted:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
		}
		wire.Type = "response.completed"
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "completed", Usage: encodeResponsesUsage(*event.Usage)}
		encoder.terminal = true
	case llmprotocol.EventResponseFailed:
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		wire.Type = "response.failed"
		wire.Error = &responsesErrorWire{Code: event.Error.Code, Message: event.Error.Message}
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Status: "failed", Error: wire.Error}
		encoder.terminal = true
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		return nil, nil, nil
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) encodeResponsesTextDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	kind := llmprotocol.ContentText
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		kind = llmprotocol.ContentRefusal
	}
	encoder.contentKinds[event.ItemIndex] = kind
	builder := encoder.contentText[event.ItemIndex]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.contentText[event.ItemIndex] = builder
	}
	builder.WriteString(event.Delta)
	var newCitations []llmprotocol.Citation
	annotationBase := len(encoder.contentCitations[event.ItemIndex])
	if event.Content != nil && len(event.Content.Citations) > 0 {
		newCitations = event.Content.Citations
		encoder.contentCitations[event.ItemIndex] = append(encoder.contentCitations[event.ItemIndex], newCitations...)
	}
	frames, err := encoder.startResponsesContent(event, kind)
	if err != nil {
		return nil, nil, err
	}
	if event.Delta != "" {
		wire := responsesEventWire{
			Type: responsesTextDeltaType(kind), Sequence: encoder.nextWireSequence(),
			ItemID: event.ItemID, OutputIndex: event.ItemIndex, Delta: event.Delta,
		}
		frame, err := encodeSSE(wire.Type, wire)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, frame)
	}
	for offset, annotation := range encodeResponsesAnnotations(newCitations) {
		index := annotationBase + offset
		wire := responsesEventWire{
			Type: "response.output_text.annotation.added", Sequence: encoder.nextWireSequence(),
			ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(),
			AnnotationIndex: &index, Annotation: &annotation,
		}
		frame, err := encodeSSE(wire.Type, wire)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil, nil
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesItem(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	wire := responsesEventWire{
		Type: "response.output_item.done", ItemID: event.ItemID, OutputIndex: event.ItemIndex,
		Item: &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "completed"},
	}
	if event.ToolCall != nil {
		wire.Item.Type, wire.Item.Role = "function_call", ""
		wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
	} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		wire.Item.Type, wire.Item.Role = "reasoning", ""
	} else {
		kind := encoder.contentKinds[event.ItemIndex]
		if kind == "" {
			kind = llmprotocol.ContentText
		}
		frames, err := encoder.completeResponsesContent(event, kind)
		if err != nil {
			return nil, nil, err
		}
		text := ""
		if builder := encoder.contentText[event.ItemIndex]; builder != nil {
			text = builder.String()
		}
		content, err := json.Marshal([]responsesContentWire{responsesContentPart(kind, text, encoder.contentCitations[event.ItemIndex])})
		if err != nil {
			return nil, nil, err
		}
		wire.Item.Content = content
		wire.Sequence = encoder.nextWireSequence()
		done, err := encodeSSE(wire.Type, wire)
		return append(frames, done), nil, err
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) nextWireSequence() uint64 {
	encoder.wireSequence++
	return encoder.wireSequence
}

func responsesContentIndex() *int {
	value := 0
	return &value
}

func responsesContentPart(kind llmprotocol.ContentKind, text string, citations []llmprotocol.Citation) responsesContentWire {
	if kind == llmprotocol.ContentRefusal {
		return responsesContentWire{Type: "refusal", Refusal: text}
	}
	return responsesContentWire{Type: "output_text", Text: text, Annotations: encodeResponsesAnnotations(citations)}
}

func responsesTextDeltaType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.delta"
	}
	return "response.output_text.delta"
}

func responsesTextDoneType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.done"
	}
	return "response.output_text.done"
}

func (encoder *responsesStreamEncoder) startResponsesContent(
	event llmprotocol.Event,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	if encoder.contentStarted[event.ItemIndex] {
		return nil, nil
	}
	encoder.contentStarted[event.ItemIndex] = true
	wire := responsesEventWire{
		Type: "response.content_part.added", Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(),
	}
	part := responsesContentPart(kind, "", nil)
	wire.Part = &part
	frame, err := encodeSSE(wire.Type, wire)
	if err != nil {
		return nil, err
	}
	return [][]byte{frame}, nil
}

func (encoder *responsesStreamEncoder) completeResponsesContent(
	event llmprotocol.Event,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	frames, err := encoder.startResponsesContent(event, kind)
	if err != nil {
		return nil, err
	}
	text := ""
	if builder := encoder.contentText[event.ItemIndex]; builder != nil {
		text = builder.String()
	}
	done := responsesEventWire{
		Type: responsesTextDoneType(kind), Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(), Text: text,
	}
	doneFrame, err := encodeSSE(done.Type, done)
	if err != nil {
		return nil, err
	}
	part := responsesContentPart(kind, text, encoder.contentCitations[event.ItemIndex])
	partDone := responsesEventWire{
		Type: "response.content_part.done", Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(), Part: &part,
	}
	partFrame, err := encodeSSE(partDone.Type, partDone)
	if err != nil {
		return nil, err
	}
	return append(frames, doneFrame, partFrame), nil
}

func (encoder *responsesStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "stream ended before completion", reason)
	wire := responsesEventWire{Type: "response.failed", Error: &responsesErrorWire{Code: protocolError.Code, Message: protocolError.Message}}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}
