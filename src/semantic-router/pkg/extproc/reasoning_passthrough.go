package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// extractReasoningContentFromMessages scans the raw request body's "messages"
// array and captures the reasoning_content value from every assistant message
// that carries one. The returned map is keyed by the message's zero-based
// index in the array; values are the raw JSON representation (string, null,
// or empty string — all of which some thinking-model endpoints require to be
// round-tripped verbatim).
//
// If the request has no messages or none of them carry reasoning_content, the
// returned map is nil, making the downstream restore a no-op.
//
// This must be called on the original (pre-SDK-parse) bytes because
// json.Unmarshal into the OpenAI SDK struct silently drops
// reasoning_content — the SDK does not model this DeepSeek extension field.
func extractReasoningContentFromMessages(body []byte) map[int]json.RawMessage {
	messages := gjson.GetBytes(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return nil
	}

	var captured map[int]json.RawMessage

	var idx int
	messages.ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		if role != "assistant" {
			idx++
			return true
		}

		rc := msg.Get("reasoning_content")
		if !rc.Exists() {
			idx++
			return true
		}

		if captured == nil {
			captured = make(map[int]json.RawMessage, 4)
		}
		// Preserve the exact JSON value (string with quotes, null literal, etc.)
		captured[idx] = json.RawMessage(rc.Raw)

		idx++
		return true
	})

	return captured
}

// restoreReasoningContentToMessages re-injects previously captured
// reasoning_content values into the serialized request body. Each entry in
// captured maps a message array index to the raw JSON value that was extracted
// before the SDK round-trip.
//
// If captured is nil or empty the input is returned unmodified (zero-copy).
func restoreReasoningContentToMessages(body []byte, captured map[int]json.RawMessage) ([]byte, error) {
	if len(captured) == 0 {
		return body, nil
	}

	var err error
	for idx, raw := range captured {
		path := fmt.Sprintf("messages.%d.reasoning_content", idx)

		// Detect the type of raw value to use the correct sjson call.
		// json.RawMessage can be a quoted string, a null literal, etc.
		var value interface{}
		if err := json.Unmarshal(raw, &value); err != nil {
			// Fallback: inject as raw string (shouldn't happen with valid JSON).
			value = string(raw)
		}

		body, err = sjson.SetBytes(body, path, value)
		if err != nil {
			return nil, fmt.Errorf("restoring reasoning_content at messages[%d]: %w", idx, err)
		}
	}

	return body, nil
}
