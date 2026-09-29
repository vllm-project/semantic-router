package protocolcodec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Captured from Ollama 0.34.3 with qwen3:8b: a tool call returned with
// "content":"" beside the reasoning and the tool call.
func TestOllamaEmptyContentBesideToolCallIsNotAnAnswer(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "providers", "ollama-chat-tool-call-out.json"))
	if err != nil {
		t.Fatal(err)
	}
	engine := NewBuiltinEngine()
	messages, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(messages.Body), `"text":""`) {
		t.Fatalf("Messages response carries an empty text block: %s", messages.Body)
	}
	if thinking, toolUse := strings.Index(string(messages.Body), `"type":"thinking"`), strings.Index(string(messages.Body), `"type":"tool_use"`); thinking < 0 || toolUse < thinking {
		t.Fatalf("want thinking followed by tool_use: %s", messages.Body)
	}
	responses, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(responses.Body), `"type":"message"`) {
		t.Fatalf("Responses output carries an empty message item: %s", responses.Body)
	}
}

func TestEmptyChatAnswerKeepsItsTextPart(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
	response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || len(response.Output[0].Content) != 1 || response.Output[0].Content[0].Kind != llmprotocol.ContentText {
		t.Fatalf("an empty answer lost its text part: %+v", response.Output)
	}
}
