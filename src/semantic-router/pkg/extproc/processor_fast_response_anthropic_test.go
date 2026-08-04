package extproc

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCreateAnthropicFastResponseNonStreaming(t *testing.T) {
	response := createAnthropicFastResponse(
		&RequestContext{VSRSelectedModel: "model-a"},
		"policy response",
		"metadata-route",
	)
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("missing immediate response")
	}
	var message struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(immediate.Body, &message); err != nil {
		t.Fatalf("decode Anthropic response: %v", err)
	}
	if message.Type != "message" || message.Role != "assistant" {
		t.Fatalf("message envelope = %#v", message)
	}
	if len(message.Content) != 1 ||
		message.Content[0].Type != "text" ||
		message.Content[0].Text != "policy response" {
		t.Fatalf("message content = %#v", message.Content)
	}
}

func TestCreateAnthropicFastResponseStreaming(t *testing.T) {
	response := createAnthropicFastResponse(
		&RequestContext{
			VSRSelectedModel:        "model-a",
			ExpectStreamingResponse: true,
		},
		"policy response",
		"metadata-route",
	)
	body := string(response.GetImmediateResponse().Body)
	for _, expected := range []string{
		"event: message_start",
		"event: content_block_delta",
		`"text":"policy response","type":"text_delta"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, expected) {
			t.Fatalf("stream body missing %q:\n%s", expected, body)
		}
	}
	if strings.Contains(body, "chat.completion.chunk") {
		t.Fatal("Anthropic stream leaked OpenAI event shape")
	}
}
