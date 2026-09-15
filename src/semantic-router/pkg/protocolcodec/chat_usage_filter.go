package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ChatUsageStreamFilter removes Router-requested accounting evidence from a
// same-format Chat stream when the public client did not opt into usage. It
// preserves every other JSON field, including provider extensions, and is
// independent from neutral semantic decoding used for accounting.
type ChatUsageStreamFilter struct {
	framer          sseFramer
	frames          int
	pendingTerminal []byte
	failure         error
	finalized       bool
}

func NewChatUsageStreamFilter(limit int) *ChatUsageStreamFilter {
	return &ChatUsageStreamFilter{framer: newSSEFramer(limit)}
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
		filtered, keep, hasData, terminal, err := filterChatUsageFrame(frame, filter.framer.limit, filter.frames == 0)
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

func filterChatUsageFrame(frame []byte, limit int, first bool) ([]byte, bool, bool, bool, error) {
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
	usage, hasUsage := object["usage"]
	if !hasUsage {
		return frame, true, true, false, nil
	}
	var choices []json.RawMessage
	if rawChoices, exists := object["choices"]; exists && !bytes.Equal(bytes.TrimSpace(rawChoices), []byte("null")) {
		if err := json.Unmarshal(rawChoices, &choices); err != nil {
			return nil, false, true, false, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
		}
	}
	if len(choices) == 0 && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
		return nil, false, true, false, nil
	}
	delete(object, "usage")
	filtered, err := encodeSSE(parsed.Event, object)
	return filtered, err == nil, true, false, err
}
