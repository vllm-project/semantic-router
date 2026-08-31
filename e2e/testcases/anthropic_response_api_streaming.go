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
	pkgtestcases.Register("anthropic-response-api-buffered", pkgtestcases.TestCase{
		Description: "POST /v1/responses over an Anthropic backend returns a buffered Responses object",
		Tags:        []string{"response-api", "anthropic", "protocol-codec"},
		Fn:          testAnthropicResponseAPIBuffered,
	})
	pkgtestcases.Register("anthropic-response-api-streaming", pkgtestcases.TestCase{
		Description: "POST /v1/responses stream:true against an api_format:anthropic backend returns Responses API SSE events",
		Tags:        []string{"response-api", "streaming", "sse", "anthropic"},
		Fn:          testAnthropicResponseAPIStreaming,
	})
	pkgtestcases.Register("anthropic-chat-completions-streaming", pkgtestcases.TestCase{
		Description: "POST /v1/chat/completions stream:true over an Anthropic backend returns Chat Completions SSE",
		Tags:        []string{"chat-completions", "streaming", "sse", "anthropic", "protocol-codec"},
		Fn:          testAnthropicChatCompletionsStreaming,
	})
}

func testAnthropicResponseAPIBuffered(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	body, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", map[string]any{
		"model": "MoM", "input": "Say hello in a few words.", "store": false,
	}, false)
	if err != nil {
		return err
	}
	var response fixtures.ResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.Object != "response" || response.Status != "completed" || len(response.Output) == 0 || strings.TrimSpace(response.OutputText) == "" {
		return fmt.Errorf("invalid buffered Responses object: %s", truncateString(string(body), 600))
	}
	return nil
}

// testAnthropicResponseAPIStreaming pins the /v1/responses streaming contract
// on the Anthropic-format backend cell. The upstream produces Anthropic
// Messages SSE; the router must translate the whole stream into Response API
// events. The shared validator also rejects leaked chat.completion.chunk
// frames and the raw [DONE] sentinel: the router's Anthropic streaming
// handler uses chat.completion.chunk as its intermediate representation,
// and this case exists to guarantee that representation never reaches a
// /v1/responses client.
func testAnthropicResponseAPIStreaming(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API streaming over the anthropic-shim backend")
	}

	result, err := requestResponseAPIStreamingSSE(ctx, client, opts, "MoM", "", "Say hello in a few words.")
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status":       result.statusCode,
			"content_type": result.contentType,
			"bytes":        len(result.body),
		})
	}

	if err := validateResponseAPIStreamingSSEResponse(result); err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Test] Anthropic-backend Response API streaming passed: bytes=%d\n", len(result.body))
	}
	return nil
}

func testAnthropicChatCompletionsStreaming(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	body, err := sendProtocolMatrixRequest(ctx, session, "/v1/chat/completions", map[string]any{
		"model":    "MoM",
		"messages": []map[string]string{{"role": "user", "content": "Say hello."}},
		"stream":   true,
	}, true)
	if err != nil {
		return err
	}
	stream := string(body)
	if !strings.Contains(stream, "chat.completion.chunk") || !strings.Contains(stream, "data: [DONE]") {
		return fmt.Errorf("invalid Chat Completions stream: %s", truncateString(stream, 600))
	}
	if strings.Contains(stream, "event: message_start") || strings.Contains(stream, "response.output_text.delta") {
		return fmt.Errorf("Chat Completions stream leaked the backend protocol: %s", truncateString(stream, 600))
	}
	return nil
}
