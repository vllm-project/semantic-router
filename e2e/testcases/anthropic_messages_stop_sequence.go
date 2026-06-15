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

// testAnthropicMessagesStopSequence asserts that the outbound emitter maps
// the upstream finish_reason to "stop_sequence" when the request carried
// stop_sequences and the model's output triggered one.
//
// The system prompt directs the tiny Qwen model to emit "STOP" verbatim,
// which — if followed — causes llama-server to set finish_reason=stop and
// return the stop token in stop_reason. The shim passes this through; the
// router's mapOpenAIFinishReasonToAnthropic must then label the response
// stop_reason as "stop_sequence" (not "end_turn").
//
// NOTE: This test depends on the tiny Qwen2.5-0.5B model in the
// anthropic-shim profile following the "say STOP exactly" instruction. If
// the model does not reliably emit the sentinel in CI, prefer replacing
// "STOP" with a sentinel the model emits unconditionally (e.g. the EOS
// token) over adding retry loops or sleeps.
//
// TODO: If this test flakes due to model instruction-following variability,
// swap the stop_sequences value for a string the model outputs in all
// completions (e.g. a fixed suffix in the system prompt) rather than
// introducing any retry or timing-based workaround.
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
		Model:         "MoM",
		MaxTokens:     50,
		StopSequences: []string{"STOP"},
		System:        "You must end every response with the word STOP in capital letters, on its own line.",
		Messages: []anthropicMessage{
			{Role: "user", Content: "Please respond and end with STOP."},
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
				"the model may not have emitted the sentinel; "+
				"if this flakes in CI replace the stop string with "+
				"a sentinel the model emits unconditionally",
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
