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
	Code            *string                  `json:"code,omitempty"`
	Message         string                   `json:"message,omitempty"`
	Param           *string                  `json:"param,omitempty"`
}

type responsesTransportErrorEventWire struct {
	Type     string  `json:"type"`
	Code     *string `json:"code"`
	Message  string  `json:"message"`
	Param    *string `json:"param"`
	Sequence uint64  `json:"sequence_number"`
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
		return decoder.decodeResponsesDoneFrame()
	}
	var wire responsesEventWire
	if err := decodeProviderWire(parsed.Data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = parsed.Event
	}
	return decoder.decodeResponsesEvent(wire, frame)
}

func (decoder *responsesStreamDecoder) decodeResponsesDoneFrame() ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	// A Responses terminal event is authoritative. Some compatible transports
	// append this sentinel for framing compatibility; it is not a second completion.
	if decoder.terminal {
		return nil, nil, nil
	}
	return decoder.emitResponsesEvent(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
}

func (decoder *responsesStreamDecoder) decodeResponsesEvent(
	wire responsesEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	event := llmprotocol.Event{ResponseID: decoder.context.ResponseID, Model: decoder.context.PublicModel, ItemIndex: wire.OutputIndex, ItemID: wire.ItemID, Delta: wire.Delta}
	if handled, events, diagnostics, err := decoder.decodeResponsesContentEvent(&event, wire); handled {
		return events, diagnostics, err
	}
	return decoder.decodeResponsesLifecycleEvent(event, wire, frame)
}

func (decoder *responsesStreamDecoder) decodeResponsesContentEvent(
	event *llmprotocol.Event,
	wire responsesEventWire,
) (bool, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	switch wire.Type {
	case "response.output_item.added":
		applyResponsesItemStart(event, wire)
	case "response.output_text.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta}
	case "response.output_text.annotation.added":
		if err := decoder.applyResponseAnnotation(event, wire); err != nil {
			return true, nil, nil, err
		}
	case "response.refusal.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: wire.Delta}
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		event.Type = llmprotocol.EventReasoningDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: wire.Delta}
	case "response.function_call_arguments.delta":
		decoder.applyResponsesToolDelta(event, wire)
	case "response.output_item.done":
		if err := decoder.applyCompletedResponseItem(event, wire); err != nil {
			return true, nil, nil, err
		}
	default:
		return false, nil, nil, nil
	}
	events, diagnostics, err := decoder.emitResponsesEvent(*event)
	return true, events, diagnostics, err
}

func (decoder *responsesStreamDecoder) decodeResponsesLifecycleEvent(
	event llmprotocol.Event,
	wire responsesEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	switch wire.Type {
	case "response.created", "response.queued", "response.in_progress":
		if decoder.started {
			return nil, nil, nil
		}
		applyResponsesStart(&event, wire)
	case "response.completed", "response.incomplete":
		applyResponsesCompletion(&event, wire)
	case "response.failed":
		applyResponseFailure(&event, wire)
	case "error":
		applyResponsesTransportFailure(&event, wire)
	case "response.content_part.added", "response.content_part.done", "response.output_text.done", "response.refusal.done", "response.function_call_arguments.done":
		return nil, nil, nil
	default:
		if err := decoder.applyUnknownResponsesEvent(&event, frame); err != nil {
			return nil, nil, err
		}
	}
	return decoder.emitResponsesEvent(event)
}

func applyResponsesStart(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseStarted
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
	}
}

func applyResponsesItemStart(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventOutputItemStarted
	if wire.Item == nil {
		return
	}
	event.ItemID = wire.Item.ID
	event.Role = llmprotocol.RoleAssistant
	if wire.Item.Type == "function_call" {
		event.ToolCall = &llmprotocol.ToolCall{ID: wire.Item.CallID, Name: wire.Item.Name, Arguments: wire.Item.Arguments}
	}
}

func (decoder *responsesStreamDecoder) applyResponsesToolDelta(event *llmprotocol.Event, wire responsesEventWire) {
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
}

func applyResponsesCompletion(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseCompleted
	event.StopReason = llmprotocol.StopEndTurn
	if wire.Type == "response.incomplete" {
		event.StopReason = llmprotocol.StopMaxTokens
	}
	if wire.Response != nil && wire.Response.Usage != nil {
		usage := decodeResponsesUsage(*wire.Response.Usage)
		event.Usage = &usage
	}
}

func (decoder *responsesStreamDecoder) applyUnknownResponsesEvent(event *llmprotocol.Event, frame []byte) error {
	if decoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || decoder.context.Source != decoder.context.Target {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Responses stream event is unsupported", nil)
	}
	event.Type, event.Opaque = llmprotocol.EventProviderOpaque, append([]byte(nil), frame...)
	return nil
}

func (decoder *responsesStreamDecoder) emitResponsesEvent(
	event llmprotocol.Event,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	normalized, err := decoder.next(event)
	return []llmprotocol.Event{normalized}, nil, err
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
	event.Failure = llmprotocol.FailureResponse
	event.Error = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
	upstreamError := wire.Error
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
		if upstreamError == nil {
			upstreamError = wire.Response.Error
		}
	}
	if upstreamError != nil {
		event.Error.Category = decodeProviderErrorCategory(upstreamError.Code)
		event.Error.Code, event.Error.Message = upstreamError.Code, upstreamError.Message
	}
}

func applyResponsesTransportFailure(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseFailed
	event.StopReason = llmprotocol.StopError
	event.Failure = llmprotocol.FailureTransport
	event.Error = llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil,
	)
	if wire.Code != nil {
		event.Error.Code = *wire.Code
		event.Error.Category = decodeProviderErrorCategory(*wire.Code)
	}
	if wire.Message != "" {
		event.Error.Message = wire.Message
	}
	if wire.Param != nil {
		event.Error.Parameter = *wire.Param
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
	if frames, diagnostics, handled, directErr := encoder.encodeDirectResponsesEvent(event); handled {
		return frames, diagnostics, directErr
	}
	wire, diagnostics, err := encoder.responsesWireForEvent(event)
	if err != nil || wire.Type == "" {
		return nil, diagnostics, err
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, diagnostics, err
}

func (encoder *responsesStreamEncoder) encodeDirectResponsesEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, bool, error) {
	switch event.Type {
	case llmprotocol.EventOutputTextDelta:
		frames, diagnostics, err := encoder.encodeResponsesTextDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventReasoningDelta:
		frames, diagnostics, err := encoder.encodeResponsesReasoningDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventOutputItemCompleted:
		frames, diagnostics, err := encoder.encodeCompletedResponsesItem(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventUsageUpdated:
		return nil, nil, true, nil
	case llmprotocol.EventProviderOpaque:
		frames, err := encoder.encodeResponsesOpaque(event)
		return frames, nil, true, err
	case llmprotocol.EventResponseFailed:
		if event.Failure == llmprotocol.FailureResponse {
			return nil, nil, false, nil
		}
		frames, err := encoder.encodeResponsesTransportFailure(event)
		return frames, nil, true, err
	default:
		return nil, nil, false, nil
	}
}

func (encoder *responsesStreamEncoder) responsesWireForEvent(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	base := responsesEventWire{ItemID: event.ItemID, OutputIndex: event.ItemIndex, Delta: event.Delta}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		base.Type = "response.created"
		base.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "in_progress"}
		return base, nil, nil
	case llmprotocol.EventOutputItemStarted:
		return encoder.responsesItemStartWire(event), nil, nil
	case llmprotocol.EventToolCallDelta:
		return responsesToolDeltaWire(event)
	case llmprotocol.EventResponseCompleted:
		return encoder.responsesCompletionWire(event)
	case llmprotocol.EventResponseFailed:
		return encoder.responsesFailureWire(event)
	default:
		return responsesEventWire{}, nil, nil
	}
}

func (encoder *responsesStreamEncoder) responsesItemStartWire(event llmprotocol.Event) responsesEventWire {
	wire := responsesEventWire{
		Type: "response.output_item.added", ItemID: event.ItemID, OutputIndex: event.ItemIndex,
		Item: &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "in_progress"},
	}
	if event.ToolCall != nil {
		wire.Item.Type, wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = "function_call", event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
		wire.Item.Role = ""
	} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		wire.Item.Type, wire.Item.Role = "reasoning", ""
	} else if event.Content != nil {
		encoder.contentKinds[event.ItemIndex] = event.Content.Kind
	}
	return wire
}

func responsesToolDeltaWire(event llmprotocol.Event) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.ToolCall == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	return responsesEventWire{
		Type: "response.function_call_arguments.delta", ItemID: event.ToolCall.ID,
		OutputIndex: event.ItemIndex, Name: event.ToolCall.Name, Delta: event.ToolCall.Arguments,
	}, nil, nil
}

func (encoder *responsesStreamEncoder) responsesCompletionWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Usage == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
	}
	encoder.terminal = true
	return responsesEventWire{
		Type:     "response.completed",
		Response: &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "completed", Usage: encodeResponsesUsage(*event.Usage)},
	}, nil, nil
}

func (encoder *responsesStreamEncoder) responsesFailureWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Error == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	return responsesEventWire{
		Type: "response.failed",
		Response: &responsesResponseWire{
			ID: event.ResponseID, Object: "response", Model: event.Model, Status: "failed",
			Error: &responsesErrorWire{Code: event.Error.Code, Message: event.Error.Message},
		},
	}, nil, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesTransportFailure(event llmprotocol.Event) ([][]byte, error) {
	if event.Error == nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	frame, err := encoder.encodeTransportError(event.Error)
	return [][]byte{frame}, err
}

func (encoder *responsesStreamEncoder) encodeResponsesReasoningDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if event.Content != nil && event.Content.Signature != "" {
		if err := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Responses cannot represent a signed reasoning delta"); err != nil {
			return nil, diagnostics, err
		}
	}
	wire := responsesEventWire{
		Type: "response.reasoning_text.delta", Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, Delta: event.Delta,
	}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, diagnostics, err
}

func (encoder *responsesStreamEncoder) encodeResponsesOpaque(event llmprotocol.Event) ([][]byte, error) {
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil
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
	wire, contentItem := responsesCompletedItemWire(event)
	if contentItem {
		return encoder.encodeCompletedResponsesContent(event, wire)
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func responsesCompletedItemWire(event llmprotocol.Event) (responsesEventWire, bool) {
	wire := responsesEventWire{
		Type: "response.output_item.done", ItemID: event.ItemID, OutputIndex: event.ItemIndex,
		Item: &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "completed"},
	}
	if event.ToolCall != nil {
		wire.Item.Type, wire.Item.Role = "function_call", ""
		wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
		return wire, false
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		wire.Item.Type, wire.Item.Role = "reasoning", ""
		return wire, false
	}
	return wire, true
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesContent(
	event llmprotocol.Event,
	wire responsesEventWire,
) ([][]byte, llmprotocol.Diagnostics, error) {
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
	frame, err := encoder.encodeTransportError(protocolError)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) encodeTransportError(protocolError *llmprotocol.ProtocolError) ([]byte, error) {
	wire := responsesTransportErrorEventWire{
		Type: "error", Code: optionalString(protocolError.Code), Message: protocolError.Message,
		Param: optionalString(protocolError.Parameter), Sequence: encoder.nextWireSequence(),
	}
	return encodeSSE(wire.Type, wire)
}
