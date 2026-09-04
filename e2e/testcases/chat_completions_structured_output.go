package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// structuredOutputProbeFormat mirrors the reproduction in issue #3024: a
// strict json_schema response_format sent through the auto-routing model.
// The simulator backend does not enforce schemas, so this contract guards the
// transport behavior: the request must route successfully (no 4xx/5xx from
// the response_format handling) and return a well-formed completion. Byte
// fidelity of the forwarded response_format is asserted by unit tests in
// src/semantic-router/pkg/extproc.
const structuredOutputProbeFormat = `{
	"type": "json_schema",
	"json_schema": {
		"name": "probe",
		"strict": true,
		"schema": {
			"type": "object",
			"properties": {"zqx_answer": {"type": "string"}},
			"required": ["zqx_answer"],
			"additionalProperties": false
		}
	}
}`

func init() {
	pkgtestcases.Register("chat-completions-structured-output", pkgtestcases.TestCase{
		Description: "Send a json_schema response_format request through the auto model and verify it routes successfully",
		Tags:        []string{"llm", "functional"},
		Fn:          testChatCompletionsStructuredOutput,
	})
}

func testChatCompletionsStructuredOutput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing structured output response_format through auto routing")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	temperature := 0.0
	chatClient := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "MoM",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Write one sentence about mountains."},
		},
		ResponseFormat: json.RawMessage(structuredOutputProbeFormat),
		Temperature:    &temperature,
		MaxTokens:      100,
	}, nil)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(resp.Body, &completion); err != nil {
		return fmt.Errorf("response is not a chat completion: %w (body: %s)", err, string(resp.Body))
	}
	if len(completion.Choices) == 0 {
		return fmt.Errorf("completion has no choices: %s", string(resp.Body))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":     resp.StatusCode,
			"response_length": len(resp.Body),
		})
	}
	return nil
}
