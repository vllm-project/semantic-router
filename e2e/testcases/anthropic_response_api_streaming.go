package testcases

import (
	"context"
	"fmt"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("anthropic-response-api-streaming", pkgtestcases.TestCase{
		Description: "POST /v1/responses stream:true against an api_format:anthropic backend returns Responses API SSE events",
		Tags:        []string{"response-api", "streaming", "sse", "anthropic"},
		Fn:          testAnthropicResponseAPIStreaming,
	})
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

	result, err := requestResponseAPIStreamingSSE(ctx, client, opts, "MoM", "anthropic-response-api-streaming", "Say hello in a few words.")
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
