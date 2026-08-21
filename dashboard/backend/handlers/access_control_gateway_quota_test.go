package handlers

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEstimateChatRequestTokensTracksTextInsteadOfJSONBytes(t *testing.T) {
	messages := json.RawMessage(`[{"role":"user","content":"` + strings.Repeat("hello world ", 1000) + `"}]`)
	estimated := estimateChatRequestTokens(messages, nil, 2048)
	byteReservation := int64(len(messages)) + 2048
	if estimated >= byteReservation || estimated < 2048 {
		t.Fatalf("estimateChatRequestTokens() = %d, byte reservation = %d", estimated, byteReservation)
	}
}

func TestEstimateChatRequestTokensProtectsMultilingualAndImageInputs(t *testing.T) {
	messages := json.RawMessage(`[{"role":"user","content":[{"type":"text","text":"请描述这张图片"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("a", 100_000) + `"}}]}]`)
	estimated := estimateChatRequestTokens(messages, nil, 512)
	if estimated < estimatedImageTokens+512 {
		t.Fatalf("estimateChatRequestTokens() = %d, want image and output allowance", estimated)
	}
	if estimated > estimatedImageTokens+2048 {
		t.Fatalf("estimateChatRequestTokens() = %d, encoded image payload was counted as text", estimated)
	}
}

func TestEstimateChatRequestTokensIncludesToolSchemas(t *testing.T) {
	messages := json.RawMessage(`[{"role":"user","content":"search"}]`)
	tools := json.RawMessage(`[{"type":"function","function":{"name":"web_search","description":"Search the web for current information"}}]`)
	withoutTools := estimateChatRequestTokens(messages, nil, 64)
	withTools := estimateChatRequestTokens(messages, tools, 64)
	if withTools <= withoutTools {
		t.Fatalf("tool estimate = %d, without tools = %d", withTools, withoutTools)
	}
}
