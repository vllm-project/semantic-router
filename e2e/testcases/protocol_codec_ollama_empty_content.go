package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const ollamaEmptyContentMarker = "__mock_ollama_empty_content__"

func init() {
	pkgtestcases.Register("protocol-codec-ollama-empty-content", pkgtestcases.TestCase{
		Description: "Ollama's empty content beside reasoning and tool calls never becomes a spurious client text block",
		Tags:        []string{"protocol-codec", "ollama", "anthropic", "response-api", "tools", "streaming"},
		Fn:          testProtocolCodecOllamaEmptyContent,
	})
}

func testProtocolCodecOllamaEmptyContent(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	// The fixture returns the captured Ollama combination of content="",
	// reasoning, and tool_calls. The client must see reasoning plus the tool,
	// with no empty text block before either one.
	messagesID := "ollama-empty-messages-" + uuid.NewString()
	messagesResult, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
		"model": ollamaCodecModel, "max_tokens": 64,
		"messages": []map[string]string{{"role": "user", "content": ollamaEmptyContentMarker}},
		"tools": []map[string]any{{
			"name": "lookup", "description": "Look up weather",
			"input_schema": map[string]any{"type": "object", "properties": map[string]any{"query": map[string]string{"type": "string"}}},
		}},
	}, false, map[string]string{"x-vsr-test-session-id": messagesID})
	if err != nil {
		return fmt.Errorf("buffered Ollama Messages request: %w", err)
	}
	if messagesResult.StatusCode != http.StatusOK {
		return fmt.Errorf("buffered Ollama Messages returned HTTP %d: %s", messagesResult.StatusCode,
			truncateString(string(messagesResult.Body), 600))
	}
	var message struct {
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type, Text, Thinking, ID, Name string
			Input                          json.RawMessage `json:"input"`
		} `json:"content"`
	}
	if decodeErr := json.Unmarshal(messagesResult.Body, &message); decodeErr != nil {
		return fmt.Errorf("decode buffered Ollama Messages response: %w", decodeErr)
	}
	if message.StopReason != "tool_use" || len(message.Content) != 2 ||
		message.Content[0].Type != "thinking" || message.Content[0].Thinking == "" ||
		message.Content[1].Type != "tool_use" || message.Content[1].ID != "call_mock_lookup" ||
		message.Content[1].Name != "lookup" || string(message.Content[1].Input) != `{"query":"weather"}` {
		return fmt.Errorf("ollama empty content displaced thinking or tool use: %s",
			truncateString(string(messagesResult.Body), 900))
	}
	if verificationErr := verifyProviderSimulatorRequest(ctx, provider, messagesID, "openai.chat.v1", ollamaEmptyContentMarker); verificationErr != nil {
		return fmt.Errorf("buffered Ollama provider dispatch: %w", verificationErr)
	}

	// Ollama also emits content="" in streamed tool deltas. Responses must
	// expose one function call and no empty message output item.
	responsesID := "ollama-empty-responses-" + uuid.NewString()
	responsesResult, err := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
		"model": ollamaCodecModel, "input": ollamaEmptyContentMarker, "store": false, "stream": true,
		"tools": []any{protocolLookupTool()},
	}, true, map[string]string{"x-vsr-test-session-id": responsesID})
	if err != nil {
		return fmt.Errorf("streamed Ollama Responses request: %w", err)
	}
	if responsesResult.StatusCode != http.StatusOK {
		return fmt.Errorf("streamed Ollama Responses returned HTTP %d: %s", responsesResult.StatusCode,
			truncateString(string(responsesResult.Body), 700))
	}
	if _, err := decodeResponsesFunctionCallStream(responsesResult.Body); err != nil {
		return fmt.Errorf("streamed Ollama Responses tool call: %w", err)
	}
	if strings.Contains(string(responsesResult.Body), `"type":"message"`) ||
		strings.Contains(string(responsesResult.Body), "response.output_text") {
		return fmt.Errorf("streamed Ollama tool call produced an empty text item: %s",
			truncateString(string(responsesResult.Body), 1000))
	}
	if verificationErr := verifyProviderSimulatorRequest(ctx, provider, responsesID, "openai.chat.v1", ollamaEmptyContentMarker); verificationErr != nil {
		return fmt.Errorf("streamed Ollama provider dispatch: %w", verificationErr)
	}
	return nil
}
