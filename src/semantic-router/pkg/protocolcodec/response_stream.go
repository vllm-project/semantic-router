package protocolcodec

import (
	"bytes"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// EncodeResponseStream renders one validated neutral response as a complete
// public event stream. It is the only path synthetic producers and cache
// replay use; neither needs to manufacture an intermediate provider payload.
func (engine *Engine) EncodeResponseStream(
	format llmprotocol.WireFormat,
	response llmprotocol.Response,
	context llmprotocol.StreamContext,
) ([]byte, llmprotocol.Diagnostics, error) {
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return nil, nil, err
	}
	if len(response.Alternatives) != 0 {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"stream_alternatives_unsupported",
			"streaming multiple response alternatives is unsupported",
			nil,
		)
	}
	pair, encodeResponseStreamErr := engine.codec(format)
	if encodeResponseStreamErr != nil {
		return nil, nil, encodeResponseStreamErr
	}
	if err := llmprotocol.RequireCapabilities(
		format,
		pair.buffered.Capabilities(),
		llmprotocol.RequiredResponseCapabilities(response),
	); err != nil {
		return nil, nil, err
	}
	context.PublicModel = response.Model
	context.ResponseID = response.ID
	encoder, encodeResponseStreamErr := engine.NewEventStreamEncoder(format, context)
	if encodeResponseStreamErr != nil {
		return nil, nil, encodeResponseStreamErr
	}
	events, encodeResponseStreamErr := neutralResponseEvents(response)
	if encodeResponseStreamErr != nil {
		return nil, nil, encodeResponseStreamErr
	}
	var body bytes.Buffer
	var diagnostics llmprotocol.Diagnostics
	for _, event := range events {
		frames, eventDiagnostics, pushErr := encoder.Push(event)
		diagnostics = appendDiagnostics(diagnostics, eventDiagnostics, engine.policy.Limits.Diagnostics)
		for _, frame := range frames {
			body.Write(frame)
		}
		if pushErr != nil {
			_, finalDiagnostics, _ := encoder.Finalize(pushErr)
			diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.policy.Limits.Diagnostics)
			return nil, diagnostics, pushErr
		}
	}
	frames, finalDiagnostics, encodeResponseStreamErr := encoder.Finalize(nil)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.policy.Limits.Diagnostics)
	for _, frame := range frames {
		body.Write(frame)
	}
	return body.Bytes(), diagnostics, encodeResponseStreamErr
}

func neutralResponseEvents(response llmprotocol.Response) ([]llmprotocol.Event, error) {
	if response.Error != nil {
		failed := llmprotocol.Event{
			Type: llmprotocol.EventResponseFailed, ResponseID: response.ID,
			Model: response.Model, StopReason: llmprotocol.StopError, Error: response.Error,
			Failure: llmprotocol.FailureResponse,
		}
		if response.Usage.State != "" {
			usage := response.Usage
			failed.Usage = &usage
		}
		return []llmprotocol.Event{failed}, nil
	}
	events := []llmprotocol.Event{{
		Type:       llmprotocol.EventResponseStarted,
		ResponseID: response.ID,
		Model:      response.Model,
	}}
	itemIndex := 0
	for outputIndex, output := range response.Output {
		groups := groupResponseStreamContent(output.Content)
		for groupIndex, group := range groups {
			context := responseEventContext{
				response: response, output: output, outputIndex: outputIndex,
				groupIndex: groupIndex, groupCount: len(groups), itemIndex: itemIndex,
			}
			contentEvents, err := neutralContentGroupEvents(context, group)
			if err != nil {
				return nil, err
			}
			events = append(events, contentEvents...)
			itemIndex++
		}
	}
	usage := response.Usage
	events = append(events, llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, ResponseID: response.ID,
		Model: response.Model, StopReason: response.StopReason,
		MatchedStopSequence: response.MatchedStopSequence, Usage: &usage,
	})
	return events, nil
}

type responseEventContext struct {
	response    llmprotocol.Response
	output      llmprotocol.OutputItem
	outputIndex int
	groupIndex  int
	groupCount  int
	itemIndex   int
}

type responseStreamContentGroup struct {
	family   llmprotocol.ContentKind
	contents []llmprotocol.Content
}

func groupResponseStreamContent(contents []llmprotocol.Content) []responseStreamContentGroup {
	if len(contents) == 0 {
		return []responseStreamContentGroup{{family: llmprotocol.ContentText}}
	}
	groups := make([]responseStreamContentGroup, 0, len(contents))
	for _, content := range contents {
		family := responseStreamContentFamily(content.Kind)
		if family == llmprotocol.ContentToolCall || len(groups) == 0 || groups[len(groups)-1].family != family {
			groups = append(groups, responseStreamContentGroup{family: family})
		}
		groups[len(groups)-1].contents = append(groups[len(groups)-1].contents, content)
	}
	return groups
}

func responseStreamContentFamily(kind llmprotocol.ContentKind) llmprotocol.ContentKind {
	switch kind {
	case llmprotocol.ContentText, llmprotocol.ContentRefusal:
		return llmprotocol.ContentText
	default:
		return kind
	}
}

func neutralContentGroupEvents(
	context responseEventContext,
	group responseStreamContentGroup,
) ([]llmprotocol.Event, error) {
	itemID := context.output.ID
	if context.groupCount > 1 {
		itemID = llmprotocol.StableID(context.output.ID, fmt.Sprint(context.groupIndex))
	}
	if itemID == "" {
		itemID = llmprotocol.StableID(context.response.ID, fmt.Sprint(context.outputIndex), fmt.Sprint(context.groupIndex))
	}
	started := llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ResponseID: context.response.ID,
		Model: context.response.Model, ItemIndex: context.itemIndex, ItemID: itemID, Role: context.output.Role,
	}
	completed := started
	completed.Type = llmprotocol.EventOutputItemCompleted
	completed.StopReason = context.response.StopReason
	if group.family == llmprotocol.ContentToolCall {
		if len(group.contents) != 1 {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorInternal,
				"stream_tool_group_invalid",
				"streaming tool calls require one output item per call",
				nil,
			)
		}
		return neutralToolCallEvents(started, completed, group.contents[0])
	}

	events := []llmprotocol.Event{started}
	for contentIndex, content := range group.contents {
		var contentEvents []llmprotocol.Event
		switch content.Kind {
		case llmprotocol.ContentText, llmprotocol.ContentRefusal:
			contentEvents = neutralTextDeltaEvents(started, content, contentIndex)
		case llmprotocol.ContentReasoning:
			contentEvents = neutralReasoningDeltaEvents(started, content, contentIndex)
		default:
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				"stream_content_unsupported",
				fmt.Sprintf("streaming output content %q is unsupported", content.Kind),
				nil,
			)
		}
		events = append(events, contentEvents...)
	}
	return append(events, completed), nil
}

func neutralTextDeltaEvents(
	started llmprotocol.Event,
	content llmprotocol.Content,
	contentIndex int,
) []llmprotocol.Event {
	delta := started
	delta.Type = llmprotocol.EventOutputTextDelta
	delta.ContentIndex = contentIndex
	delta.Delta = content.Text
	delta.Content = &content
	return []llmprotocol.Event{delta}
}

func neutralReasoningDeltaEvents(
	started llmprotocol.Event,
	content llmprotocol.Content,
	contentIndex int,
) []llmprotocol.Event {
	events := make([]llmprotocol.Event, 0, 2)
	if content.Text != "" {
		reasoning := content
		reasoning.Signature = ""
		delta := started
		delta.Type = llmprotocol.EventReasoningDelta
		delta.ContentIndex = contentIndex
		delta.Delta = content.Text
		delta.Content = &reasoning
		events = append(events, delta)
	}
	if content.Signature != "" {
		signature := llmprotocol.Content{
			Kind: content.Kind, Signature: content.Signature, Reasoning: content.Reasoning,
		}
		delta := started
		delta.Type = llmprotocol.EventReasoningDelta
		delta.ContentIndex = contentIndex
		delta.Content = &signature
		events = append(events, delta)
	}
	if len(events) == 0 {
		delta := started
		delta.Type = llmprotocol.EventReasoningDelta
		delta.ContentIndex = contentIndex
		delta.Content = &content
		events = append(events, delta)
	}
	return events
}

func neutralToolCallEvents(started, completed llmprotocol.Event, content llmprotocol.Content) ([]llmprotocol.Event, error) {
	if content.ToolCall == nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_call_missing", "neutral tool call is missing", nil)
	}
	call := *content.ToolCall
	started.ToolCall = &llmprotocol.ToolCall{ID: call.ID, Name: call.Name}
	completed.ToolCall = &call
	delta := started
	delta.Type = llmprotocol.EventToolCallDelta
	delta.ToolCall = &call
	return []llmprotocol.Event{started, delta, completed}, nil
}
