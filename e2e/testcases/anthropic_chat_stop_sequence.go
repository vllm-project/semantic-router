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

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-chat-stop-sequence", pkgtestcases.TestCase{
		Description: "A Chat client whose request stops on a sequence receives finish_reason=stop over the Anthropic backend instead of an unsupported-capability error (issue #4892)",
		Tags:        []string{"chat-completions", "stop-reason", "anthropic", "streaming", "functional"},
		Fn:          testAnthropicChatStopSequence,
	})
}

// testAnthropicChatStopSequence pins the projected terminal-reason contract for
// chat clients on the Anthropic-format backend: the request carries `stop`, the
// provider fixture truncates on it, and the reply must complete with the
// generic `finish_reason: "stop"` — buffered and streamed — rather than fail
// with unsupported_capability now that the matched string has no Chat field.
func testAnthropicChatStopSequence(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing chat finish_reason=stop projection on /v1/chat/completions")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	payload := map[string]any{
		"model":       "vllm-sr/auto",
		"max_tokens":  50,
		"temperature": 0,
		// A stop that lands mid-canned-text keeps the case independent of the
		// whitespace-only matched-stop edge, which the stream provider
		// validation still rejects (#4909).
		"stop": []string{" from"},
		"messages": []map[string]string{{
			"role":    "user",
			"content": "Answer directly with a short phrase of at least three words.",
		}},
	}

	bufferedBody, err := postChatCompletions(ctx, localPort, payload)
	if err != nil {
		return err
	}
	var buffered struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if decodeErr := json.Unmarshal(bufferedBody, &buffered); decodeErr != nil {
		return fmt.Errorf("unmarshal buffered response: %w (body=%s)", decodeErr, truncateString(string(bufferedBody), 300))
	}
	if len(buffered.Choices) == 0 || buffered.Choices[0].FinishReason != "stop" {
		return fmt.Errorf("expected finish_reason=stop, got %s", truncateString(string(bufferedBody), 600))
	}

	payload["stream"] = true
	streamBody, err := postChatCompletions(ctx, localPort, payload)
	if err != nil {
		return err
	}
	stream := string(streamBody)
	if strings.Contains(stream, `"error"`) {
		return fmt.Errorf("streamed reply ended on an error frame: %s", truncateString(stream, 600))
	}
	finalFinish := ""
	for _, line := range strings.Split(stream, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason any `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if finish, ok := choice.FinishReason.(string); ok {
				finalFinish = finish
			}
		}
	}
	if finalFinish != "stop" {
		return fmt.Errorf("expected the streamed terminal finish_reason=stop, got %q: %s", finalFinish, truncateString(stream, 600))
	}
	return nil
}

func postChatCompletions(ctx context.Context, localPort string, payload map[string]any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, truncateString(string(raw), 300))
	}
	return raw, nil
}
