package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("response-api-streaming-sse", pkgtestcases.TestCase{
		Description: "POST /v1/responses stream:true returns Responses API SSE events",
		Tags:        []string{"response-api", "streaming", "sse"},
		Fn:          testResponseAPIStreamingSSE,
	})
}

func testResponseAPIStreamingSSE(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API streaming SSE: POST /v1/responses stream:true")
	}

	result, err := requestResponseAPIStreamingSSE(ctx, client, opts)
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
		fmt.Printf("[Test] Response API streaming SSE passed: bytes=%d\n", len(result.body))
	}
	return nil
}

type responseAPIStreamingSSEResult struct {
	statusCode  int
	contentType string
	body        []byte
}

func requestResponseAPIStreamingSSE(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (responseAPIStreamingSSEResult, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return responseAPIStreamingSSEResult{}, err
	}
	defer session.Close()

	body := map[string]interface{}{
		"model":  "openai/gpt-oss-20b",
		"input":  "Stream this response through the Responses API.",
		"stream": true,
		"store":  false,
		"metadata": map[string]string{
			"test": "response-api-streaming-sse",
		},
	}
	rawBody, err := json.Marshal(body)
	if err != nil {
		return responseAPIStreamingSSEResult{}, fmt.Errorf("marshal streaming response-api request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		session.BaseURL()+"/v1/responses",
		bytes.NewReader(rawBody),
	)
	if err != nil {
		return responseAPIStreamingSSEResult{}, fmt.Errorf("create streaming response-api request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := session.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return responseAPIStreamingSSEResult{}, fmt.Errorf("send streaming response-api request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return responseAPIStreamingSSEResult{}, fmt.Errorf("read streaming response-api body: %w", err)
	}

	return responseAPIStreamingSSEResult{
		statusCode:  resp.StatusCode,
		contentType: resp.Header.Get("Content-Type"),
		body:        responseBody,
	}, nil
}

func validateResponseAPIStreamingSSEResponse(result responseAPIStreamingSSEResult) error {
	stream := string(result.body)

	if result.statusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", result.statusCode, truncateString(stream, 500))
	}
	if !strings.Contains(result.contentType, "text/event-stream") {
		return fmt.Errorf("expected text/event-stream content-type, got %q", result.contentType)
	}

	return validateResponseAPIStreamingSSEBody(stream)
}

func validateResponseAPIStreamingSSEBody(stream string) error {
	requiredEvents := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.content_part.added",
		"event: response.output_text.delta",
		"event: response.output_text.done",
		"event: response.content_part.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	for _, event := range requiredEvents {
		if !strings.Contains(stream, event) {
			return fmt.Errorf("missing Responses API SSE event %q in stream: %s", event, truncateString(stream, 800))
		}
	}

	forbiddenFragments := []struct {
		value string
		error string
	}{
		{
			value: "chat.completion.chunk",
			error: "stream leaked upstream Chat Completions chunk instead of Responses API events",
		},
		{
			value: "data: [DONE]",
			error: "stream leaked raw upstream [DONE] sentinel instead of response.completed",
		},
		{
			value: "sequence_number",
			error: "stream included non-Responses API sequence_number field",
		},
	}
	for _, fragment := range forbiddenFragments {
		if strings.Contains(stream, fragment.value) {
			return fmt.Errorf("%s: %s", fragment.error, truncateString(stream, 800))
		}
	}
	if !strings.Contains(stream, `"annotations":[]`) {
		return fmt.Errorf("stream missing output_text annotations array: %s", truncateString(stream, 800))
	}

	return nil
}
