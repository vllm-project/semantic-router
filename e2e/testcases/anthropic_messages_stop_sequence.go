package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("anthropic-messages-stop-sequence", pkgtestcases.TestCase{
		Description: "Verify stop_reason=stop_sequence when stop_sequences is set and model triggers it (anthropic-shim profile)",
		Tags:        []string{"anthropic", "stop-reason", "functional"},
		Fn:          testAnthropicMessagesStopSequence,
	})
}

// testAnthropicMessagesStopSequence asserts that the response carries
// stop_reason "stop_sequence" (not "end_turn") when the request's
// stop_sequences triggered generation stop, exercising the router's
// Anthropic outbound translation of stop_sequences end to end.
//
// The stop sequences are single vowels, which any English completion emits
// within its first few tokens regardless of instruction-following — the
// tiny Qwen2.5-0.5B model in this profile cannot be trusted to emit an
// agreed sentinel like "STOP" (in CI it answered in 3 tokens and hit EOS
// first). With unconditional stop strings, a failure means the
// stop_sequences never reached llama-server or the stop_reason mapping is
// wrong, not that the model ignored the prompt.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesStopSequence(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing stop_sequence assertion on /v1/messages")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	body := stopSequenceRequestBody{
		Model:     "MoM",
		MaxTokens: 50,
		// Single vowels: emitted unconditionally by any English output, so
		// triggering does not depend on the 0.5B model following
		// instructions. Four entries stay within the OpenAI stop-list cap
		// that the router's internal representation is translated through.
		StopSequences: []string{"e", "a", "i", "o"},
		System:        "You are a helpful assistant.",
		Messages: []anthropicMessage{
			{Role: "user", Content: "Introduce yourself in one short English sentence."},
		},
	}

	jsonData, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/messages", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}
	var parsed anthropicStopResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf(
			"unmarshal response: %w (body=%s)", err, truncateString(string(raw), 200),
		)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code": resp.StatusCode,
			"stop_reason": parsed.StopReason,
		})
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s",
			resp.StatusCode, truncateString(string(raw), 200))
	}

	if parsed.StopReason != "stop_sequence" {
		return fmt.Errorf(
			"expected stop_reason=stop_sequence, got %q — "+
				"the stop strings are unconditional vowels, so either "+
				"stop_sequences were dropped in the router's outbound "+
				"translation or the upstream stop_reason mapping is wrong",
			parsed.StopReason,
		)
	}

	return nil
}

// anthropicStopResponse is the minimal parse target for stop-sequence
// assertions. Only stop_reason is needed; using a local type avoids a
// dependency on anthropicCacheResponse from the sibling cache-cycle test.
type anthropicStopResponse struct {
	StopReason string `json:"stop_reason"`
}

// stopSequenceRequestBody is the POST /v1/messages payload for the
// stop-sequence assertion. stop_sequences is what triggers the mapping in
// mapOpenAIFinishReasonToAnthropic; system uses a string (not array) so
// cache_control synthesis is not involved.
type stopSequenceRequestBody struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	StopSequences []string           `json:"stop_sequences"`
	System        string             `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
}
