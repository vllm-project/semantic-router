package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ChatUsageStreamFilter frames a same-format Chat stream before exposing it to
// the client. It removes Router-requested accounting evidence when the client
// did not opt into usage, and provider decorations that the neutral codec
// reports as dropped. Undecorated frames retain their original bytes.
type ChatUsageStreamFilter struct {
	framer          sseFramer
	stripUsage      bool
	seenFinish      map[int]bool
	frames          int
	pendingTerminal []byte
	failure         error
	finalized       bool
}

func NewChatUsageStreamFilter(limit int) *ChatUsageStreamFilter {
	return NewChatPublicStreamFilter(limit, false)
}

// NewChatPublicStreamFilter preserves opted-in usage while removing provider
// decorations that cannot appear in the public Chat response contract.
func NewChatPublicStreamFilter(limit int, includeUsage bool) *ChatUsageStreamFilter {
	return &ChatUsageStreamFilter{
		framer: newSSEFramer(limit), stripUsage: !includeUsage,
		seenFinish: make(map[int]bool),
	}
}

func (filter *ChatUsageStreamFilter) Push(chunk []byte) ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.failure != nil {
		return nil, filter.failure
	}
	if filter.finalized {
		return nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already finalized", nil)
	}
	frames, err := filter.framer.Push(chunk)
	if err != nil {
		return nil, filter.poison(err)
	}
	return filter.filterFrames(frames)
}

func (filter *ChatUsageStreamFilter) Finalize() ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.finalized {
		return nil, nil
	}
	filter.finalized = true
	if filter.failure != nil {
		return nil, filter.failure
	}
	frames, err := filter.framer.Finalize()
	if err != nil {
		return nil, filter.poison(err)
	}
	output, err := filter.filterFrames(frames)
	if err != nil {
		return nil, err
	}
	output = append(output, filter.pendingTerminal...)
	filter.pendingTerminal = nil
	return output, nil
}

func (filter *ChatUsageStreamFilter) filterFrames(frames [][]byte) ([]byte, error) {
	var output bytes.Buffer
	for _, frame := range frames {
		filtered, keep, hasData, terminal, err := filterChatUsageFrame(
			frame, filter.framer.limit, filter.frames == 0, filter.stripUsage, filter.seenFinish,
		)
		filter.frames++
		if err != nil {
			return nil, filter.poison(err)
		}
		if len(filter.pendingTerminal) != 0 && hasData {
			return nil, filter.poison(invalidProviderResponse(
				"stream_event_after_terminal",
				"Chat stream emitted data after its terminal sentinel",
			))
		}
		if terminal {
			filter.pendingTerminal = append(filter.pendingTerminal[:0], filtered...)
			continue
		}
		if keep {
			output.Write(filtered)
		}
	}
	return output.Bytes(), nil
}

func (filter *ChatUsageStreamFilter) poison(err error) error {
	if err != nil && filter.failure == nil {
		filter.failure = err
	}
	filter.pendingTerminal = nil
	return filter.failure
}

func filterChatUsageFrame(frame []byte, limit int, first, stripUsage bool, seenFinish map[int]bool) ([]byte, bool, bool, bool, error) {
	parsed, err := parseSSEFrameAtPosition(frame, limit, first)
	if err != nil {
		return nil, false, false, false, err
	}
	if !parsed.HasData {
		return frame, true, false, false, nil
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		return frame, true, true, true, nil
	}
	var object map[string]json.RawMessage
	if err := decodeProviderWire(parsed.Data, &object, llmprotocol.DefaultPolicy()); err != nil {
		return nil, false, true, false, err
	}
	changed := false
	if _, present := object["provider"]; present {
		delete(object, "provider")
		changed = true
	}
	var choices []json.RawMessage
	if rawChoices, exists := object["choices"]; exists && !bytes.Equal(bytes.TrimSpace(rawChoices), []byte("null")) {
		if err := json.Unmarshal(rawChoices, &choices); err != nil {
			return nil, false, true, false, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
		}
	}
	for index, rawChoice := range choices {
		choice, removed, choiceErr := removeChatJSONFields(rawChoice, "native_finish_reason")
		if choiceErr != nil {
			return nil, false, true, false, choiceErr
		}
		if removed {
			choices[index] = choice
			changed = true
		}
	}
	var removedRepeatedFinish bool
	choices, removedRepeatedFinish, err = removeRepeatedChatFinishChoices(choices, seenFinish)
	if err != nil {
		return nil, false, true, false, err
	}
	changed = changed || removedRepeatedFinish
	if changed && (len(choices) > 0 || removedRepeatedFinish) {
		encoded, marshalErr := json.Marshal(choices)
		if marshalErr != nil {
			return nil, false, true, false, marshalErr
		}
		object["choices"] = encoded
	}
	usage, hasUsage := object["usage"]
	if removedRepeatedFinish && len(choices) == 0 && (!hasUsage || bytes.Equal(bytes.TrimSpace(usage), []byte("null"))) {
		return nil, false, true, false, nil
	}
	if hasUsage && stripUsage {
		if len(choices) == 0 && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
			return nil, false, true, false, nil
		}
		delete(object, "usage")
		changed = true
	} else if hasUsage && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
		usage, changed, err = stripOpenRouterUsageDecorations(usage, changed)
		if err != nil {
			return nil, false, true, false, err
		}
		object["usage"] = usage
	}
	if !changed {
		return frame, true, true, false, nil
	}
	filtered, err := encodeSSE(parsed.Event, object)
	return filtered, err == nil, true, false, err
}

// Some providers send an extra content-free finish chunk after an earlier
// finish for the same choice. Keep the first terminal, and retain a repeated
// chunk's usage as a usage-only frame if the client requested accounting.
func removeRepeatedChatFinishChoices(choices []json.RawMessage, seen map[int]bool) ([]json.RawMessage, bool, error) {
	if len(choices) == 0 {
		return choices, false, nil
	}
	kept := make([]json.RawMessage, 0, len(choices))
	removed := false
	for _, raw := range choices {
		var choice struct {
			Index        *int            `json:"index"`
			FinishReason *string         `json:"finish_reason"`
			Delta        json.RawMessage `json:"delta"`
		}
		if err := json.Unmarshal(raw, &choice); err != nil {
			return nil, false, err
		}
		if choice.Index == nil || choice.FinishReason == nil {
			kept = append(kept, raw)
			continue
		}
		index := *choice.Index
		if seen[index] && chatFinishChoiceIsContentFree(raw, choice.Delta) {
			removed = true
			continue
		}
		seen[index] = true
		kept = append(kept, raw)
	}
	return kept, removed, nil
}

func chatFinishChoiceIsContentFree(rawChoice, rawDelta json.RawMessage) bool {
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(rawChoice, &choice); err != nil {
		return false
	}
	for field, value := range choice {
		if field != "index" && field != "finish_reason" && field != "delta" &&
			(field != "logprobs" || !bytes.Equal(bytes.TrimSpace(value), []byte("null"))) {
			return false
		}
	}
	if len(rawDelta) == 0 || bytes.Equal(bytes.TrimSpace(rawDelta), []byte("null")) {
		return true
	}
	var delta map[string]json.RawMessage
	if err := json.Unmarshal(rawDelta, &delta); err != nil {
		return false
	}
	for field, value := range delta {
		if field != "content" ||
			!bytes.Equal(bytes.TrimSpace(value), []byte(`""`)) &&
				!bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	return true
}

func removeChatJSONFields(raw json.RawMessage, fields ...string) (json.RawMessage, bool, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, false, err
	}
	changed := false
	for _, field := range fields {
		if _, present := object[field]; present {
			delete(object, field)
			changed = true
		}
	}
	if !changed {
		return raw, false, nil
	}
	encoded, err := json.Marshal(object)
	return encoded, true, err
}

func stripOpenRouterUsageDecorations(raw json.RawMessage, changed bool) (json.RawMessage, bool, error) {
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw, &usage); err != nil {
		return nil, false, err
	}
	for _, field := range []string{"cost", "is_byok", "cost_details", "server_tool_use"} {
		if _, present := usage[field]; present {
			delete(usage, field)
			changed = true
		}
	}
	for _, detail := range []struct{ field, nested string }{
		{"prompt_tokens_details", "video_tokens"},
		{"completion_tokens_details", "image_tokens"},
	} {
		value, present := usage[detail.field]
		if !present || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			continue
		}
		filtered, removed, err := removeChatJSONFields(value, detail.nested)
		if err != nil {
			return nil, false, err
		}
		if removed {
			usage[detail.field] = filtered
			changed = true
		}
	}
	if !changed {
		return raw, false, nil
	}
	encoded, err := json.Marshal(usage)
	return encoded, true, err
}
