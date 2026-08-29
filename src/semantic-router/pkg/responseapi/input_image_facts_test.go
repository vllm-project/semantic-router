package responseapi

import (
	"encoding/json"
	"testing"
)

func TestCountNativeImageInputs_StringInput(t *testing.T) {
	if got := CountNativeImageInputs(json.RawMessage(`"describe this"`), nil); got != 0 {
		t.Errorf("expected 0 for string input, got %d", got)
	}
}

func TestCountNativeImageInputs_EmptyInput(t *testing.T) {
	if got := CountNativeImageInputs(nil, nil); got != 0 {
		t.Errorf("expected 0 for empty input, got %d", got)
	}
}

func TestCountNativeImageInputs_URLImage(t *testing.T) {
	input := json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [
			{"type": "input_text", "text": "what is this?"},
			{"type": "input_image", "image_url": "https://example.com/a.png"}
		]
	}]`)
	if got := CountNativeImageInputs(input, nil); got != 1 {
		t.Errorf("expected 1 for URL image, got %d", got)
	}
}

func TestCountNativeImageInputs_FileIDImage(t *testing.T) {
	input := json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [
			{"type": "input_text", "text": "what is this?"},
			{"type": "input_image", "file_id": "file-abc123"}
		]
	}]`)
	if got := CountNativeImageInputs(input, nil); got != 1 {
		t.Errorf("expected 1 for file_id image, got %d", got)
	}
}

func TestCountNativeImageInputs_FileDataImage(t *testing.T) {
	input := json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [{"type": "input_image", "file_data": "aGVsbG8="}]
	}]`)
	if got := CountNativeImageInputs(input, nil); got != 1 {
		t.Errorf("expected 1 for file_data image, got %d", got)
	}
}

func TestCountNativeImageInputs_MixedParts(t *testing.T) {
	input := json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [
			{"type": "input_text", "text": "compare these"},
			{"type": "input_image", "image_url": "https://example.com/a.png"},
			{"type": "input_image", "file_id": "file-abc123"},
			{"type": "input_file", "file_id": "file-not-an-image"},
			{"type": "input_image"}
		]
	}]`)
	if got := CountNativeImageInputs(input, nil); got != 2 {
		t.Errorf("expected 2 (URL + file_id, not input_file or empty input_image), got %d", got)
	}
}

func TestCountNativeImageInputs_MultipleItems(t *testing.T) {
	input := json.RawMessage(`[
		{"type": "message", "role": "user", "content": [{"type": "input_image", "file_id": "file-1"}]},
		{"type": "message", "role": "assistant", "content": "sure"},
		{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/b.png"}]}
	]`)
	if got := CountNativeImageInputs(input, nil); got != 2 {
		t.Errorf("expected 2 across items, got %d", got)
	}
}

func TestCountNativeImageInputs_History(t *testing.T) {
	history := []*StoredResponse{
		nil,
		{
			Input: []InputItem{{
				Type:    ItemTypeMessage,
				Role:    RoleUser,
				Content: json.RawMessage(`[{"type": "input_image", "file_id": "file-old"}]`),
			}},
		},
	}
	input := json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [{"type": "input_text", "text": "and this one?"}]
	}]`)
	if got := CountNativeImageInputs(input, history); got != 1 {
		t.Errorf("expected 1 from history, got %d", got)
	}
}

func TestCountNativeImageInputs_MalformedContent(t *testing.T) {
	input := json.RawMessage(`[{"type": "message", "role": "user", "content": {"not": "parts"}}]`)
	if got := CountNativeImageInputs(input, nil); got != 0 {
		t.Errorf("expected 0 for malformed content, got %d", got)
	}
}
