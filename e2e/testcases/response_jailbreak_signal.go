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

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("response-jailbreak-streaming-passthrough", pkgtestcases.TestCase{
		Description: "Verify a streamed response is scored by the response-direction jailbreak rule and still delivered in full, because a streamed response is observed and never enforced",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak", "streaming"},
		Fn:          testResponseJailbreakStreamingPassthrough,
	})
}

// responseJailbreakSignalRule is the response-direction rule the profile
// declares, and the series name its extraction is counted under.
const responseJailbreakSignalRule = "unsafe_completion"

// testResponseJailbreakStreamingPassthrough pins the streaming contract: the
// response-direction rule scores a streamed response once the stream ends and
// records the observation, and nothing enforces it. The same content through
// the same decision that blocks it buffered is delivered in full, with no block
// and no warning header, because its bytes were with the client before the
// answer existed as a whole.
//
// Both halves are asserted together. The delivery on its own would also pass
// with the answer never scored, which is what this case pinned before the rule
// reached the streaming path.
func testResponseJailbreakStreamingPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a streamed response is scored and still passes through")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open metrics session: %w", err)
	}
	defer metricsSession.Close()

	scoredBefore, err := readSignalExtractionCount(ctx, metricsSession, "jailbreak", responseJailbreakSignalRule)
	if err != nil {
		return fmt.Errorf("read %s before the streamed request: %w", signalExtractionMetric, err)
	}

	prompt := responseJailbreakPrompt(responseJailbreakBlockProbe, responseJailbreakPhrase)
	resp, err := sendResponseJailbreakStreamingRequest(ctx, localPort, prompt)
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	defer resp.Body.Close()

	streamBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read stream: %w", err)
	}
	content, frames := chatStreamedContent(streamBody)
	warnings := resp.Header.Get(responseWarningsHeader)

	scoredAfter, err := readSignalExtractionCount(ctx, metricsSession, "jailbreak", responseJailbreakSignalRule)
	if err != nil {
		return fmt.Errorf("read %s after the streamed request: %w", signalExtractionMetric, err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   resp.StatusCode,
			"frames":        frames,
			"content_bytes": len(content),
			"warnings":      warnings,
			"scored_before": scoredBefore,
			"scored_after":  scoredAfter,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] status=%d frames=%d content_bytes=%d warnings=%q scored=%v->%v\n",
			resp.StatusCode, frames, len(content), warnings, scoredBefore, scoredAfter)
	}

	if resp.StatusCode != 200 {
		return fmt.Errorf("expected the streamed response to be delivered, got %d: %s", resp.StatusCode, truncateString(string(streamBody), 400))
	}
	if !strings.Contains(content, responseJailbreakPhrase) {
		return fmt.Errorf("the streamed content did not arrive in full (%d frames, %d bytes): the stream was cut or altered", frames, len(content))
	}
	if scoredAfter < scoredBefore+1 {
		return fmt.Errorf("the streamed response was not scored by rule %q: %s went %v to %v",
			responseJailbreakSignalRule, signalExtractionMetric, scoredBefore, scoredAfter)
	}
	if strings.Contains(warnings, responseJailbreakWarningCode) {
		return fmt.Errorf("a streamed response carried %s = %q; the bytes were already delivered, so the plugin cannot have acted on the detection",
			responseWarningsHeader, warnings)
	}

	return nil
}

func sendResponseJailbreakStreamingRequest(ctx context.Context, localPort, prompt string) (*http.Response, error) {
	requestBody := map[string]interface{}{
		"model":  "MoM",
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}
	url := fmt.Sprintf("http://localhost:%s%s", localPort, localChatCompletionsPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("x-vsr-debug", "true")
	return (&http.Client{Timeout: 60 * time.Second}).Do(req)
}

// chatStreamedContent joins the content deltas of an OpenAI chat completion SSE
// stream and reports how many frames carried them. mock-vllm streams the answer
// in fixed-size chunks, so the text under test can straddle two frames and has
// to be reassembled.
func chatStreamedContent(streamBody []byte) (string, int) {
	var content strings.Builder
	frames := 0
	for _, data := range protocolSSEDataFrames(streamBody) {
		if data == "[DONE]" {
			continue
		}
		frames++
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			content.WriteString(choice.Delta.Content)
		}
	}
	return content.String(), frames
}
