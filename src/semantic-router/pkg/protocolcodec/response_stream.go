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
		return []llmprotocol.Event{{
			Type: llmprotocol.EventResponseFailed, ResponseID: response.ID,
			Model: response.Model, StopReason: llmprotocol.StopError, Error: response.Error,
		}}, nil
	}
	events := []llmprotocol.Event{{
		Type:       llmprotocol.EventResponseStarted,
		ResponseID: response.ID,
		Model:      response.Model,
	}}
	itemIndex := 0
	for outputIndex, output := range response.Output {
		for contentIndex := range output.Content {
			content := output.Content[contentIndex]
			itemID := output.ID
			if len(output.Content) > 1 {
				itemID = llmprotocol.StableID(output.ID, fmt.Sprint(contentIndex))
			}
			if itemID == "" {
				itemID = llmprotocol.StableID(response.ID, fmt.Sprint(outputIndex), fmt.Sprint(contentIndex))
			}
			started := llmprotocol.Event{
				Type: llmprotocol.EventOutputItemStarted, ResponseID: response.ID,
				Model: response.Model, ItemIndex: itemIndex, ItemID: itemID, Role: output.Role,
			}
			completed := started
			completed.Type = llmprotocol.EventOutputItemCompleted
			completed.StopReason = response.StopReason
			switch content.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal:
				started.Content = &llmprotocol.Content{Kind: content.Kind}
				completed.Content = &content
				events = append(events, started, llmprotocol.Event{
					Type: llmprotocol.EventOutputTextDelta, ResponseID: response.ID,
					Model: response.Model, ItemIndex: itemIndex, ItemID: itemID,
					Role: output.Role, Delta: content.Text, Content: &content,
				}, completed)
			case llmprotocol.ContentReasoning:
				started.Content = &llmprotocol.Content{Kind: content.Kind}
				completed.Content = &content
				events = append(events, started)
				if content.Text != "" {
					reasoning := content
					reasoning.Signature = ""
					events = append(events, llmprotocol.Event{
						Type: llmprotocol.EventReasoningDelta, ResponseID: response.ID,
						Model: response.Model, ItemIndex: itemIndex, ItemID: itemID,
						Role: output.Role, Delta: content.Text, Content: &reasoning,
					})
				}
				if content.Signature != "" {
					signature := llmprotocol.Content{Kind: content.Kind, Signature: content.Signature}
					events = append(events, llmprotocol.Event{
						Type: llmprotocol.EventReasoningDelta, ResponseID: response.ID,
						Model: response.Model, ItemIndex: itemIndex, ItemID: itemID,
						Role: output.Role, Content: &signature,
					})
				}
				events = append(events, completed)
			case llmprotocol.ContentToolCall:
				if content.ToolCall == nil {
					return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_call_missing", "neutral tool call is missing", nil)
				}
				call := *content.ToolCall
				started.ToolCall = &llmprotocol.ToolCall{ID: call.ID, Name: call.Name}
				completed.ToolCall = &call
				events = append(events, started, llmprotocol.Event{
					Type: llmprotocol.EventToolCallDelta, ResponseID: response.ID,
					Model: response.Model, ItemIndex: itemIndex, ItemID: itemID,
					Role: output.Role, ToolCall: &call,
				}, completed)
			default:
				return nil, llmprotocol.NewError(
					llmprotocol.ErrorUnsupportedFeature,
					"stream_content_unsupported",
					fmt.Sprintf("streaming output content %q is unsupported", content.Kind),
					nil,
				)
			}
			itemIndex++
		}
	}
	usage := response.Usage
	events = append(events, llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, ResponseID: response.ID,
		Model: response.Model, StopReason: response.StopReason, Usage: &usage,
	})
	return events, nil
}
