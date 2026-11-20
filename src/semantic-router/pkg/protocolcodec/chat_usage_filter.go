package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ChatUsageStreamFilter frames a same-format Chat stream before exposing it to
// the client. It removes Router-requested accounting evidence when the client
// did not opt into usage, and provider decorations that the neutral codec
// reports as dropped. When the client opted into usage, a chunk without usage
// gains "usage": null and keeps its other bytes; other undecorated frames
// retain their original bytes. A provider error event ends the public stream:
// the filter drops it and every later frame, and UpstreamFailed reports it.
type ChatUsageStreamFilter struct {
	framer          sseFramer
	stripUsage      bool
	seenFinish      map[int]bool
	frames          int
	pendingTerminal []byte
	failure         error
	finalized       bool
	upstreamFailed  bool
	// outputLimit is the output token limit the backend received, 0 for none.
	// A frame that ends the choice for its tool calls is held with the frames
	// after it until usage tells whether the output reached the limit, which
	// the Router then reports as the stop, as the neutral stream decodes it.
	outputLimit int64
	held        *heldChatFinish
}

// heldChatFinish holds a frame that ends the choice for its tool calls and the
// frames after it.
type heldChatFinish struct {
	frames [][]byte
	first  bool // the finish frame opened the stream
}

// chatFrameEvidence is what a frame tells the output limit check.
type chatFrameEvidence struct {
	toolCallFinish bool   // the frame ends choice 0 for its tool calls
	outputTokens   *int64 // the completion tokens of the frame's usage
}

// errChatUpstreamErrorFrame marks a provider error event. The filter drops it
// and every frame after it; the Router sends its own error frame instead.
var errChatUpstreamErrorFrame = errors.New("chat stream carried a provider error event")

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

// UpstreamFailed reports whether the stream carried a provider error event.
// The filter dropped that event, so when this is true the caller must send the
// Router's error frame in its place.
func (filter *ChatUsageStreamFilter) UpstreamFailed() bool {
	return filter != nil && filter.upstreamFailed
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
	held, err := filter.releaseHeld(false)
	if err != nil {
		return nil, filter.poison(err)
	}
	output = append(output, held...)
	output = append(output, filter.pendingTerminal...)
	filter.pendingTerminal = nil
	return output, nil
}

func (filter *ChatUsageStreamFilter) filterFrames(frames [][]byte) ([]byte, error) {
	var output bytes.Buffer
	for _, frame := range frames {
		if filter.upstreamFailed {
			continue
		}
		first := filter.frames == 0
		var evidence *chatFrameEvidence
		if filter.outputLimit > 0 {
			evidence = &chatFrameEvidence{}
		}
		filtered, keep, hasData, terminal, err := filterChatUsageFrame(
			frame, filter.framer.limit, first, filter.stripUsage, filter.seenFinish, evidence,
		)
		filter.frames++
		if errors.Is(err, errChatUpstreamErrorFrame) {
			if len(filter.pendingTerminal) == 0 {
				filter.upstreamFailed = true
				held, heldErr := filter.releaseHeld(false)
				if heldErr != nil {
					return nil, filter.poison(heldErr)
				}
				output.Write(held)
				continue
			}
			// After [DONE], an error event is data after the terminal, as before.
			err = nil
		}
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
			held, heldErr := filter.releaseHeld(false)
			if heldErr != nil {
				return nil, filter.poison(heldErr)
			}
			output.Write(held)
			filter.pendingTerminal = append(filter.pendingTerminal[:0], filtered...)
			continue
		}
		passed, err := filter.pass(filtered, keep, first, evidence)
		if err != nil {
			return nil, filter.poison(err)
		}
		output.Write(passed)
	}
	return output.Bytes(), nil
}

// pass returns what the client gets now of a filtered frame. A frame that
// ends the choice for its tool calls waits, with the frames after it, for the
// usage that tells whether the output reached the output limit.
func (filter *ChatUsageStreamFilter) pass(frame []byte, keep, first bool, evidence *chatFrameEvidence) ([]byte, error) {
	if evidence == nil || filter.held == nil && (!keep || !evidence.toolCallFinish) {
		if !keep {
			return nil, nil
		}
		return frame, nil
	}
	if filter.held == nil {
		filter.held = &heldChatFinish{first: first}
	}
	if keep {
		filter.held.frames = append(filter.held.frames, frame)
	}
	if evidence.outputTokens == nil {
		return nil, nil
	}
	return filter.releaseHeld(*evidence.outputTokens >= filter.outputLimit)
}

// releaseHeld returns the held frames, with the finish frame reporting the
// output limit stop when atLimit.
func (filter *ChatUsageStreamFilter) releaseHeld(atLimit bool) ([]byte, error) {
	held := filter.held
	if held == nil {
		return nil, nil
	}
	filter.held = nil
	if atLimit {
		finish, err := chatFinishAtOutputLimit(held.frames[0], filter.framer.limit, held.first)
		if err != nil {
			return nil, err
		}
		held.frames[0] = finish
	}
	return bytes.Join(held.frames, nil), nil
}

func (filter *ChatUsageStreamFilter) poison(err error) error {
	if err != nil && filter.failure == nil {
		filter.failure = err
	}
	filter.pendingTerminal = nil
	filter.held = nil
	return filter.failure
}

// chatFinishAtOutputLimit rewrites the tool-call finish of choice 0 in a
// frame as the output limit stop.
func chatFinishAtOutputLimit(frame []byte, limit int, first bool) ([]byte, error) {
	parsed, err := parseSSEFrameAtPosition(frame, limit, first)
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err = json.Unmarshal(parsed.Data, &object); err != nil {
		return nil, err
	}
	var choices []map[string]json.RawMessage
	if err = json.Unmarshal(object["choices"], &choices); err != nil {
		return nil, err
	}
	stop, err := json.Marshal(encodeChatStop(llmprotocol.StopMaxTokens))
	if err != nil {
		return nil, err
	}
	for _, choice := range choices {
		var index int
		var reason string
		if json.Unmarshal(choice["index"], &index) == nil && index == 0 &&
			json.Unmarshal(choice["finish_reason"], &reason) == nil && decodeChatStop(reason) == llmprotocol.StopToolCall {
			choice["finish_reason"] = stop
		}
	}
	if object["choices"], err = json.Marshal(choices); err != nil {
		return nil, err
	}
	return encodeSSE(parsed.Event, object)
}

// chatChoicesEndForToolCalls reports choices that end choice 0 for its tool
// calls.
func chatChoicesEndForToolCalls(choices []json.RawMessage) bool {
	for _, raw := range choices {
		var choice struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason"`
		}
		if json.Unmarshal(raw, &choice) == nil && choice.Index == 0 && choice.FinishReason != nil &&
			decodeChatStop(*choice.FinishReason) == llmprotocol.StopToolCall {
			return true
		}
	}
	return false
}

func chatUsageCompletionTokens(raw json.RawMessage) *int64 {
	var usage map[string]json.RawMessage
	if json.Unmarshal(raw, &usage) != nil {
		return nil
	}
	// Match Chat provider decoding: a present canonical counter takes precedence.
	counter, present := usage["completion_tokens"]
	if !present {
		counter = usage["output_tokens"]
	}
	var tokens *int64
	if counter == nil || json.Unmarshal(counter, &tokens) != nil {
		return nil
	}
	return tokens
}

func filterChatUsageFrame(
	frame []byte, limit int, first, stripUsage bool, seenFinish map[int]bool, evidence *chatFrameEvidence,
) ([]byte, bool, bool, bool, error) {
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
	if raw, failed := object["error"]; failed && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, false, true, false, errChatUpstreamErrorFrame
	}
	changed, err := deleteChatServerMembers(object)
	if err != nil {
		return nil, false, true, false, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
	}
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
	if evidence != nil {
		evidence.toolCallFinish = chatChoicesEndForToolCalls(choices)
		if hasUsage {
			evidence.outputTokens = chatUsageCompletionTokens(usage)
		}
	}
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
	} else if !hasUsage && !stripUsage {
		if !changed {
			frame, err = encodeNullUsageFrame(parsed.Event, parsed.Data)
			return frame, err == nil, true, false, err
		}
		object["usage"] = json.RawMessage("null")
	}
	if !changed {
		return frame, true, true, false, nil
	}
	filtered, err := encodeSSE(parsed.Event, object)
	return filtered, err == nil, true, false, err
}

// encodeNullUsageFrame gives a chunk without usage the "usage": null that
// OpenAI sends on every chunk before the usage chunk of a stream whose client
// asked for usage, leaving the provider's members as they were.
func encodeNullUsageFrame(event string, data []byte) ([]byte, error) {
	var chunk bytes.Buffer
	if err := json.Compact(&chunk, data); err != nil {
		return nil, err
	}
	body := chunk.Bytes()
	separator := ","
	if len(body) == len("{}") {
		separator = ""
	}
	var frame bytes.Buffer
	if event != "" {
		frame.WriteString("event: " + event + "\n")
	}
	frame.WriteString("data: ")
	frame.Write(body[:len(body)-1])
	frame.WriteString(separator + `"usage":null}` + "\n\n")
	return frame.Bytes(), nil
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
