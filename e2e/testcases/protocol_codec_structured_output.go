package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("protocol-codec-chat-backend-structured-output", pkgtestcases.TestCase{
		Description: "Every client preserves JSON Schema output constraints through a Chat Completions backend",
		Tags:        []string{"protocol-codec", "structured-output", "matrix", "streaming"},
		Fn:          testProtocolCodecChatBackendStructuredOutput,
	})
	pkgtestcases.Register("protocol-codec-responses-backend-structured-output", pkgtestcases.TestCase{
		Description: "Every client preserves JSON Schema output constraints through a Responses backend",
		Tags:        []string{"protocol-codec", "structured-output", "matrix", "streaming"},
		Fn:          testProtocolCodecResponsesBackendStructuredOutput,
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-structured-output", pkgtestcases.TestCase{
		Description: "Every client preserves JSON Schema output constraints through an Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "structured-output", "anthropic", "matrix", "streaming"},
		Fn:          testProtocolCodecAnthropicBackendStructuredOutput,
	})
}

func testProtocolCodecChatBackendStructuredOutput(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecStructuredOutputMatrix(ctx, client, opts, chatBackendModel, "openai.chat.v1")
}

func testProtocolCodecResponsesBackendStructuredOutput(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecStructuredOutputMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1")
}

func testProtocolCodecAnthropicBackendStructuredOutput(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runProtocolCodecStructuredOutputMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1")
}

func runProtocolCodecStructuredOutputMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	for _, streaming := range []bool{false, true} {
		for _, clientCase := range protocolStructuredOutputClients(model, streaming) {
			body, requestErr := sendProtocolMatrixRequest(
				ctx, session, clientCase.path, clientCase.body, streaming,
			)
			if requestErr != nil {
				return fmt.Errorf("%s stream=%t: %w", clientCase.name, streaming, requestErr)
			}
			text, extractErr := extractProtocolStructuredOutputText(clientCase.path, body, streaming)
			if extractErr != nil {
				return fmt.Errorf("%s stream=%t: %w", clientCase.name, streaming, extractErr)
			}
			if verifyErr := verifyProtocolStructuredOutputEcho(text); verifyErr != nil {
				return fmt.Errorf("%s stream=%t: %w", clientCase.name, streaming, verifyErr)
			}
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"backend_format": backendFormat,
			"client_formats": 3,
			"request_modes":  2,
		})
	}
	return nil
}

type protocolStructuredOutputClient struct {
	name string
	path string
	body map[string]any
}

func protocolStructuredOutputClients(model string, streaming bool) []protocolStructuredOutputClient {
	schema := protocolStructuredOutputSchema()
	return []protocolStructuredOutputClient{
		{
			name: "chat_completions", path: "/v1/chat/completions",
			body: map[string]any{
				"model": model, "stream": streaming,
				"messages": []map[string]string{{"role": "user", "content": "Return an answer object."}},
				"response_format": map[string]any{
					"type": "json_schema",
					"json_schema": map[string]any{
						"name": "structured_output", "strict": true, "schema": schema,
					},
				},
			},
		},
		{
			name: "openai_responses", path: "/v1/responses",
			body: map[string]any{
				"model": model, "input": "Return an answer object.", "stream": streaming, "store": false,
				"text": map[string]any{"format": map[string]any{
					"type": "json_schema", "name": "structured_output", "strict": true, "schema": schema,
				}},
			},
		},
		{
			name: "anthropic_messages", path: "/v1/messages",
			body: map[string]any{
				"model": model, "max_tokens": 64, "stream": streaming,
				"messages": []map[string]string{{"role": "user", "content": "Return an answer object."}},
				"output_config": map[string]any{"format": map[string]any{
					"type": "json_schema", "schema": schema,
				}},
			},
		},
	}
}

func protocolStructuredOutputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"$defs": map[string]any{
			"citation": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"title": map[string]any{"type": "string", "minLength": 1},
					"url":   map[string]any{"type": "string", "format": "uri"},
				},
				"required":             []string{"title", "url"},
				"additionalProperties": false,
			},
		},
		"properties": map[string]any{
			"answer":     map[string]any{"type": "string", "minLength": 1},
			"confidence": map[string]any{"type": "number", "minimum": 0, "maximum": 1},
			"status":     map[string]any{"type": "string", "enum": []string{"grounded", "uncertain"}},
			"citations": map[string]any{
				"type":  "array",
				"items": map[string]any{"$ref": "#/$defs/citation"},
			},
		},
		"required":             []string{"answer", "confidence", "status", "citations"},
		"additionalProperties": false,
	}
}

func extractProtocolStructuredOutputText(path string, body []byte, streaming bool) (string, error) {
	if streaming {
		return extractProtocolStructuredOutputStreamText(path, body)
	}
	switch path {
	case "/v1/chat/completions":
		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return "", err
		}
		if len(response.Choices) != 1 {
			return "", fmt.Errorf("Chat response has %d choices: %s", len(response.Choices), truncateString(string(body), 500))
		}
		return response.Choices[0].Message.Content, nil
	case "/v1/responses":
		var response fixtures.ResponseAPIResponse
		if err := json.Unmarshal(body, &response); err != nil {
			return "", err
		}
		return response.OutputText, nil
	case "/v1/messages":
		var response anthropicMessageResponse
		if err := json.Unmarshal(body, &response); err != nil {
			return "", err
		}
		var text strings.Builder
		for _, rawContent := range response.Content {
			var content struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal(rawContent, &content); err != nil {
				return "", fmt.Errorf("decode Messages content block: %w", err)
			}
			if content.Type == "text" {
				text.WriteString(content.Text)
			}
		}
		if text.Len() == 0 {
			return "", fmt.Errorf("Messages response has no text content: %s", truncateString(string(body), 500))
		}
		return text.String(), nil
	default:
		return "", fmt.Errorf("unsupported client path %q", path)
	}
}

func extractProtocolStructuredOutputStreamText(path string, body []byte) (string, error) {
	var text strings.Builder
	for _, frame := range strings.Split(string(body), "\n\n") {
		data := ""
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data:") {
				data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				break
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		switch path {
		case "/v1/chat/completions":
			var event struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				return "", fmt.Errorf("decode Chat SSE event: %w", err)
			}
			if len(event.Choices) > 0 {
				text.WriteString(event.Choices[0].Delta.Content)
			}
		case "/v1/responses":
			var event struct {
				Type  string `json:"type"`
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				return "", fmt.Errorf("decode Responses SSE event: %w", err)
			}
			if event.Type == "response.output_text.delta" {
				text.WriteString(event.Delta)
			}
		case "/v1/messages":
			var event struct {
				Type  string `json:"type"`
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				return "", fmt.Errorf("decode Messages SSE event: %w", err)
			}
			if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
				text.WriteString(event.Delta.Text)
			}
		default:
			return "", fmt.Errorf("unsupported client path %q", path)
		}
	}
	if text.Len() == 0 {
		return "", fmt.Errorf("client stream has no output text: %s", truncateString(string(body), 800))
	}
	return text.String(), nil
}

func verifyProtocolStructuredOutputEcho(text string) error {
	var echo struct {
		Mock             string         `json:"mock"`
		StructuredOutput map[string]any `json:"structured_output"`
	}
	if err := json.Unmarshal([]byte(text), &echo); err != nil {
		return fmt.Errorf("backend echo is not JSON: %w (text=%q)", err, truncateString(text, 500))
	}
	if echo.Mock != "mock-vllm" || echo.StructuredOutput["type"] != "json_schema" {
		return fmt.Errorf("backend did not receive a JSON Schema output contract: %s", truncateString(text, 500))
	}
	schema := echo.StructuredOutput["schema"]
	if nested, ok := echo.StructuredOutput["json_schema"].(map[string]any); ok {
		schema = nested["schema"]
	}
	actual, err := json.Marshal(schema)
	if err != nil {
		return err
	}
	expected, err := json.Marshal(protocolStructuredOutputSchema())
	if err != nil {
		return err
	}
	if string(actual) != string(expected) {
		return fmt.Errorf("backend schema changed: got=%s want=%s", actual, expected)
	}
	return nil
}
