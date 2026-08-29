package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
	case "response.content_part.added", "response.content_part.done",
		"response.output_text.done", "response.refusal.done",
		"response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
		"response.reasoning_summary_text.done", "response.reasoning_text.done":
		return nil, nil, nil
	case "response.function_call_arguments.done":
		if err := decoder.validateResponsesToolDone(wire); err != nil {
			return nil, nil, err
		}
		return nil, nil, nil
	default:
		if err := decoder.applyUnknownResponsesEvent(&event, frame); err != nil {
			return nil, nil, err
		}
	}
	return decoder.emitResponsesEvent(event)
}

func (decoder *responsesStreamDecoder) validateResponsesToolDone(wire responsesEventWire) error {
	index := responsesWireOutputIndex(wire)
	call := decoder.toolCalls[index]
	if call.Name != "" && wire.Name != call.Name {
		return invalidProviderResponse("stream_tool_identity_mismatch", "Responses function-call done event changed the tool name")
	}
	if call.Name == "" {
		call.Name = wire.Name
		decoder.toolCalls[index] = call
	}
	arguments := decoder.toolArguments[index]
	if len(arguments) == 0 {
		if !isJSONObject([]byte(wire.Arguments), decoder.policy.Limits.JSONDepth) {
			return invalidProviderResponse("invalid_stream_tool_arguments", "Responses function-call done arguments must be a JSON object")
		}
		decoder.toolArguments[index] = []byte(wire.Arguments)
		decoder.toolArgumentsDone[index] = true
		return nil
	}
	if string(arguments) != wire.Arguments {
		return invalidProviderResponse("stream_tool_arguments_mismatch", "Responses function-call done arguments do not match streamed arguments")
	}
	decoder.toolArgumentsDone[index] = true
	return nil
}

func applyResponsesStart(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseStarted
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
	}
}

func (decoder *responsesStreamDecoder) applyResponsesItemStart(event *llmprotocol.Event, wire responsesEventWire) error {
	event.Type = llmprotocol.EventOutputItemStarted
	if len(wire.Item) == 0 {
		return nil
	}
	item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
	if err != nil {
		return err
	}
	if err := validateResponsesOutputItemResource(wire.Item, item, decoder.policy.Limits); err != nil {
		return err
	}
	if item.Status != "" && item.Status != "in_progress" {
		return invalidProviderResponse(
			"stream_item_status_mismatch",
			"Responses output item added event requires in_progress status",
		)
	}
	event.ItemID = item.ID
	decoder.itemTypes[responsesWireOutputIndex(wire)] = item.Type
	event.Role = llmprotocol.RoleAssistant
	if item.Type == "function_call" {
		event.ToolCall = &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}
	} else if item.Type == "reasoning" {
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	} else if item.Type == "image_generation_call" {
		event.Content = &llmprotocol.Content{
			Kind:           llmprotocol.ContentGeneratedImage,
			GeneratedImage: decodeResponsesGeneratedImage(item),
		}
	}
	return nil
}

func (decoder *responsesStreamDecoder) applyResponsesToolDelta(event *llmprotocol.Event, wire responsesEventWire) error {
	event.Type = llmprotocol.EventToolCallDelta
	call := decoder.toolCalls[responsesWireOutputIndex(wire)]
	if len(wire.Item) > 0 {
		item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
		if err != nil {
			return err
		}
		if item.CallID != "" {
			call.ID = item.CallID
		}
	}
	if wire.Name != "" {
		call.Name = wire.Name
	}
	call.Arguments = wire.Delta
	event.ToolCall = &call
	return nil
}

func applyResponsesCompletion(event *llmprotocol.Event, wire responsesEventWire) {
	event.Type = llmprotocol.EventResponseCompleted
	event.StopReason = llmprotocol.StopEndTurn
	if wire.Type == "response.incomplete" {
		event.StopReason = llmprotocol.StopUnknown
		if wire.Response != nil && wire.Response.IncompleteDetails != nil {
			switch wire.Response.IncompleteDetails.Reason {
			case "max_output_tokens":
				event.StopReason = llmprotocol.StopMaxTokens
			case "content_filter":
				event.StopReason = llmprotocol.StopContentFilter
			}
		}
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
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.Event{normalized}, nil, nil
}

func (decoder *responsesStreamDecoder) applyResponseAnnotation(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	if wire.Annotation == nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_missing", "Responses citation event is missing its annotation", nil)
	}
	outputIndex := responsesWireOutputIndex(wire)
	itemID, itemFound := decoder.itemIDs[outputIndex]
	if !itemFound || wire.ItemID == "" || wire.ItemID != itemID {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_item", "Responses citation event does not match its active output item", nil)
	}
	if wire.ContentIndex == nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_content_index", "Responses citation event is missing its content index", nil)
	}
	key := streamContentKey{item: outputIndex, content: *wire.ContentIndex}
	expected := decoder.nextAnnotationIndexes[key]
	if wire.AnnotationIndex == nil || *wire.AnnotationIndex != expected {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_index", "Responses citation indexes must be monotonic and contiguous", nil)
	}
	citations, err := decodeResponsesAnnotations([]responsesAnnotationWire{*wire.Annotation})
	if err != nil {
		return err
	}
	event.Type = llmprotocol.EventOutputTextDelta
	event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
	decoder.nextAnnotationIndexes[key] = expected + 1
	return nil
}

func (decoder *responsesStreamDecoder) applyCompletedResponseItem(
	event *llmprotocol.Event, wire responsesEventWire,
) error {
	event.Type = llmprotocol.EventOutputItemCompleted
	if len(wire.Item) == 0 {
		return nil
	}
	item, err := decoder.validateCompletedResponseItem(wire)
	if err != nil {
		return err
	}
	event.ItemID = item.ID
	if err := decoder.applyCompletedResponseItemKind(event, wire, item); err != nil {
		return err
	}
	decoder.completedOutput[responsesWireOutputIndex(wire)] = append(json.RawMessage(nil), wire.Item...)
	return nil
}

func (decoder *responsesStreamDecoder) validateCompletedResponseItem(wire responsesEventWire) (responsesItemWire, error) {
	item, err := decodeResponsesItemWire(wire.Item, decoder.policy, true)
	if err != nil {
		return responsesItemWire{}, err
	}
	if err := validateResponsesOutputItemResource(wire.Item, item, decoder.policy.Limits); err != nil {
		return responsesItemWire{}, err
	}
	if item.Type == "image_generation_call" && item.Status != "completed" && item.Status != "failed" {
		return responsesItemWire{}, invalidProviderResponse(
			"stream_item_status_mismatch", "Responses image generation item done event requires completed or failed status",
		)
	}
	if item.Type != "image_generation_call" && item.Status != "" && item.Status != "completed" && item.Status != "incomplete" {
		return responsesItemWire{}, invalidProviderResponse(
			"stream_item_status_mismatch", "Responses output item done event requires completed or incomplete status",
		)
	}
	if expected := decoder.itemTypes[responsesWireOutputIndex(wire)]; expected == "" || item.Type != expected {
		return responsesItemWire{}, invalidProviderResponse(
			"stream_item_kind_mismatch", "Responses output item completion changed its item type",
		)
	}
	return item, nil
}

func (decoder *responsesStreamDecoder) applyCompletedResponseItemKind(
	event *llmprotocol.Event,
	wire responsesEventWire,
	item responsesItemWire,
) error {
	switch item.Type {
	case "function_call":
		event.ToolCall = &llmprotocol.ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments}
	case "message":
		if decoder.itemKinds[responsesWireOutputIndex(wire)] == llmprotocol.ContentToolCall {
			return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_kind_mismatch", "upstream completed a tool item as a message", nil)
		}
	case "reasoning":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
	case "image_generation_call":
		event.Content = &llmprotocol.Content{
			Kind:           llmprotocol.ContentGeneratedImage,
			GeneratedImage: decodeResponsesGeneratedImage(item),
		}
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
	var upstreamError *responsesErrorWire
	if wire.Response != nil {
		event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
		upstreamError = wire.Response.Error
		if wire.Response.Usage != nil {
			usage := decodeResponsesUsage(*wire.Response.Usage)
			event.Usage = &usage
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
