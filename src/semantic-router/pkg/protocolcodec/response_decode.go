/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package protocolcodec

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// DecodeResponse decodes one buffered wire response into the neutral response
// contract without encoding a second provider representation.
func (engine *Engine) DecodeResponse(
	format llmprotocol.WireFormat,
	body []byte,
) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response, envelope, diagnostics, err := pair.buffered.DecodeResponse(body, engine.policy)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return response, envelope, diagnostics, err
	}
	return response, envelope, diagnostics, nil
}

// DecodeResponseStream decodes a complete wire stream into one neutral
// terminal response. It is intended for buffered orchestration such as Looper;
// forwarding paths should keep using StreamEngine to preserve backpressure.
func (engine *Engine) DecodeResponseStream(
	format llmprotocol.WireFormat,
	body []byte,
	context llmprotocol.StreamContext,
) (llmprotocol.Response, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Response{}, nil, err
	}
	if !pair.buffered.Capabilities().Supports(llmprotocol.CapabilityStreaming) {
		return llmprotocol.Response{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"streaming_unsupported",
			"wire format does not support streaming",
			nil,
		)
	}
	context.Source = format
	context.Target = format
	decoder := pair.stream.NewDecoder(context, engine.strictStreamPolicy())
	accumulator := newResponseAccumulator()
	events, diagnostics, err := decoder.Push(body)
	if applyErr := accumulator.apply(events); applyErr != nil {
		return llmprotocol.Response{}, diagnostics, applyErr
	}
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	finalEvents, finalDiagnostics, err := decoder.Finalize(nil)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.policy.Limits.Diagnostics)
	if applyErr := accumulator.apply(finalEvents); applyErr != nil {
		return llmprotocol.Response{}, diagnostics, applyErr
	}
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	response, err := accumulator.response()
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return response, diagnostics, err
	}
	return response, diagnostics, nil
}

type responseAccumulator struct {
	result   llmprotocol.Response
	items    map[int]*responseAccumulatorItem
	terminal bool
}

type responseAccumulatorItem struct {
	id       string
	role     llmprotocol.Role
	contents []llmprotocol.Content
	call     *llmprotocol.ToolCall
}

func newResponseAccumulator() *responseAccumulator {
	return &responseAccumulator{
		result: llmprotocol.Response{
			Generation: 1,
			Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		},
		items: make(map[int]*responseAccumulatorItem),
	}
}

func (accumulator *responseAccumulator) apply(events []llmprotocol.Event) error {
	for _, event := range events {
		if err := accumulator.applyEvent(event); err != nil {
			return err
		}
	}
	return nil
}

func (accumulator *responseAccumulator) applyEvent(event llmprotocol.Event) error {
	if accumulator.terminal {
		return llmprotocol.NewError(
			llmprotocol.ErrorConflict,
			"stream_terminal",
			"stream emitted an event after its terminal",
			nil,
		)
	}
	if event.ResponseID != "" {
		accumulator.result.ID = event.ResponseID
	}
	if event.Model != "" {
		accumulator.result.Model = event.Model
	}
	if event.Usage != nil {
		accumulator.result.Usage = *event.Usage
	}
	switch event.Type {
	case llmprotocol.EventResponseStarted, llmprotocol.EventUsageUpdated:
		return nil
	case llmprotocol.EventOutputItemStarted:
		if _, found := accumulator.items[event.ItemIndex]; found {
			return fmt.Errorf("neutral stream item %d started twice", event.ItemIndex)
		}
		item := &responseAccumulatorItem{id: event.ItemID, role: event.Role}
		if item.role == "" {
			item.role = llmprotocol.RoleAssistant
		}
		if event.ToolCall != nil {
			call := *event.ToolCall
			item.call = &call
		} else if event.Content != nil {
			item.contents = append(item.contents, *event.Content)
		}
		accumulator.items[event.ItemIndex] = item
		return nil
	case llmprotocol.EventOutputTextDelta:
		kind := llmprotocol.ContentText
		var citations []llmprotocol.Citation
		if event.Content != nil && event.Content.Kind != "" {
			kind = event.Content.Kind
			citations = event.Content.Citations
		}
		return accumulator.appendText(event.ItemIndex, kind, event.Delta, "", citations)
	case llmprotocol.EventReasoningDelta:
		signature := ""
		if event.Content != nil {
			signature = event.Content.Signature
		}
		return accumulator.appendText(event.ItemIndex, llmprotocol.ContentReasoning, event.Delta, signature, nil)
	case llmprotocol.EventToolCallDelta:
		return accumulator.appendToolCall(event.ItemIndex, event.ToolCall)
	case llmprotocol.EventOutputItemCompleted:
		return accumulator.completeItem(event)
	case llmprotocol.EventResponseCompleted:
		accumulator.result.StopReason = event.StopReason
		accumulator.terminal = true
		return nil
	case llmprotocol.EventResponseFailed:
		accumulator.result.Output = nil
		accumulator.result.StopReason = llmprotocol.StopError
		accumulator.result.Usage = llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
		accumulator.result.Error = event.Error
		accumulator.terminal = true
		return nil
	case llmprotocol.EventProviderOpaque:
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"opaque_stream_event",
			"provider-opaque stream events cannot enter the neutral response contract",
			nil,
		)
	default:
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unknown_stream_event",
			"stream emitted an unknown semantic event",
			nil,
		)
	}
}

func (accumulator *responseAccumulator) item(index int) (*responseAccumulatorItem, error) {
	item := accumulator.items[index]
	if item == nil {
		return nil, fmt.Errorf("neutral stream item %d is unavailable", index)
	}
	return item, nil
}

func (accumulator *responseAccumulator) appendText(
	index int,
	kind llmprotocol.ContentKind,
	text string,
	signature string,
	citations []llmprotocol.Citation,
) error {
	item, err := accumulator.item(index)
	if err != nil {
		return err
	}
	for contentIndex := range item.contents {
		content := &item.contents[contentIndex]
		if content.Kind != kind {
			continue
		}
		content.Text += text
		if signature != "" {
			content.Signature = signature
		}
		content.Citations = append(content.Citations, citations...)
		return nil
	}
	item.contents = append(item.contents, llmprotocol.Content{Kind: kind, Text: text, Signature: signature, Citations: append([]llmprotocol.Citation(nil), citations...)})
	return nil
}

func (accumulator *responseAccumulator) appendToolCall(index int, delta *llmprotocol.ToolCall) error {
	if delta == nil {
		return fmt.Errorf("neutral tool-call delta is unavailable")
	}
	item, err := accumulator.item(index)
	if err != nil {
		return err
	}
	if item.call == nil {
		item.call = &llmprotocol.ToolCall{}
	}
	if strings.TrimSpace(item.call.Arguments) == "{}" && delta.Arguments != "" {
		item.call.Arguments = ""
	}
	if delta.ID != "" {
		item.call.ID = delta.ID
	}
	if delta.Name != "" {
		item.call.Name = delta.Name
	}
	item.call.Arguments += delta.Arguments
	return nil
}

func (accumulator *responseAccumulator) completeItem(event llmprotocol.Event) error {
	item, err := accumulator.item(event.ItemIndex)
	if err != nil {
		return err
	}
	if event.ItemID != "" {
		item.id = event.ItemID
	}
	if event.Role != "" {
		item.role = event.Role
	}
	if event.ToolCall != nil {
		call := *event.ToolCall
		item.call = &call
	}
	if event.Content != nil && event.Content.Kind != "" && event.Content.Text != "" {
		for index := range item.contents {
			if item.contents[index].Kind == event.Content.Kind {
				item.contents[index] = *event.Content
				return nil
			}
		}
		item.contents = append(item.contents, *event.Content)
	}
	return nil
}

func (accumulator *responseAccumulator) response() (llmprotocol.Response, error) {
	if !accumulator.terminal {
		return llmprotocol.Response{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_incomplete",
			"upstream stream ended without a terminal response",
			nil,
		)
	}
	if accumulator.result.Error != nil {
		return accumulator.result, nil
	}
	indexes := make([]int, 0, len(accumulator.items))
	for index := range accumulator.items {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		item := accumulator.items[index]
		if len(item.contents) == 0 && item.call == nil {
			// Some wire formats emit a role-only framing item before the first
			// semantic content block. It is transport state, not an empty output.
			continue
		}
		contents := append([]llmprotocol.Content(nil), item.contents...)
		if item.call != nil {
			call := *item.call
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &call})
		}
		accumulator.result.Output = append(accumulator.result.Output, llmprotocol.OutputItem{
			ID: item.id, Role: item.role, Content: contents,
		})
	}
	return accumulator.result, nil
}
