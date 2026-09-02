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

const (
	// The compose probe selects response_jailbreak_compose_route at request
	// time, which carries no response plugin. Only the response-stage decision
	// composed of the same keyword AND the response-direction jailbreak rule
	// has a block action, so a 403 proves the composition drove it.
	responseJailbreakComposeProbe    = "vsr-response-compose-probe"
	responseJailbreakComposeDecision = "response_jailbreak_compose_block"
)

func init() {
	pkgtestcases.Register("response-jailbreak-compose-block", pkgtestcases.TestCase{
		Description: "Verify a response-stage decision composed of a request keyword and the response-direction jailbreak rule blocks the buffered response",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak"},
		Fn:          testResponseJailbreakComposeBlock,
	})
	pkgtestcases.Register("response-jailbreak-streaming-passthrough", pkgtestcases.TestCase{
		Description: "Verify a streamed response is delivered in full under the same response-stage configuration, since the response-direction rule is scored for buffered responses only",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak", "streaming"},
		Fn:          testResponseJailbreakStreamingPassthrough,
	})
}

// testResponseJailbreakComposeBlock drives the buffered response path with the
// compose probe. The request-stage decision routes the request without any
// response plugin; once the model answers, the response-direction rule matches
// and the response-stage decision - keyword AND jailbreak - selects, and its
// block action turns the response into a 403. The benign control runs the same
// probe through the same decisions, so the block has to come from the
// response observation and not from the keyword alone.
func testResponseJailbreakComposeBlock(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing a response-stage decision composed of a request keyword and the response-direction jailbreak rule")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	unsafe := responseJailbreakPrompt(responseJailbreakComposeProbe, responseJailbreakPhrase)
	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", unsafe, 60*time.Second)
	if err != nil {
		return fmt.Errorf("chat completion request failed: %w", err)
	}

	control := responseJailbreakPrompt(responseJailbreakComposeProbe, "That is the whole history of navigation.")
	controlResponse, err := sendLocalChatCompletion(ctx, localPort, "MoM", control, 60*time.Second)
	if err != nil {
		return fmt.Errorf("control chat completion request failed: %w", err)
	}

	body := string(response.Body)
	decision := response.Headers.Get(vsrSelectedDecisionHeader)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":         response.StatusCode,
			"selected_decision":   decision,
			"control_status_code": controlResponse.StatusCode,
			"control_warnings":    controlResponse.Headers.Get(responseWarningsHeader),
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] status=%d decision=%q control_status=%d\n", response.StatusCode, decision, controlResponse.StatusCode)
	}

	if response.StatusCode != 403 {
		return fmt.Errorf(
			"expected the response-stage decision %q to block with 403, got %d - the request decision has no response plugin, so nothing else can block.\n%s",
			responseJailbreakComposeDecision, response.StatusCode, formatUnexpectedChatCompletionStatus(response),
		)
	}
	if !strings.Contains(body, "jailbreak content detected") {
		return fmt.Errorf("403 body does not read as a response-jailbreak block: %s", truncateString(body, 400))
	}
	if controlResponse.StatusCode != 200 {
		return fmt.Errorf("a benign response through the same decisions was not delivered: got %d\n%s",
			controlResponse.StatusCode, formatUnexpectedChatCompletionStatus(controlResponse))
	}
	if warnings := controlResponse.Headers.Get(responseWarningsHeader); strings.Contains(warnings, responseJailbreakWarningCode) {
		return fmt.Errorf("a benign response through the same decisions warned (%s = %q)", responseWarningsHeader, warnings)
	}

	return nil
}

// testResponseJailbreakStreamingPassthrough pins the streaming contract under
// the same configuration: the response-direction rule is scored for buffered
// responses only, so a streamed response carrying the same content past the
// classifier window is delivered in full, with no block and no warning header.
// This is the current behavior, not the target; it is asserted so that scoring
// streamed responses is a deliberate change rather than a drift.
func testResponseJailbreakStreamingPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a streamed response passes through the response-stage configuration")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	prompt := responseJailbreakPrompt(responseJailbreakComposeProbe, responseJailbreakPhrase)
	resp, err := sendResponseJailbreakStreamingRequest(ctx, localPort, prompt)
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	defer resp.Body.Close()

	streamBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read stream: %w", err)
	}
	content, frames := responseJailbreakStreamedContent(streamBody)
	warnings := resp.Header.Get(responseWarningsHeader)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   resp.StatusCode,
			"frames":        frames,
			"content_bytes": len(content),
			"warnings":      warnings,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] status=%d frames=%d content_bytes=%d warnings=%q\n", resp.StatusCode, frames, len(content), warnings)
	}

	if resp.StatusCode != 200 {
		return fmt.Errorf("expected the streamed response to be delivered, got %d: %s", resp.StatusCode, truncateString(string(streamBody), 400))
	}
	if !strings.Contains(content, responseJailbreakPhrase) {
		return fmt.Errorf("the streamed content did not arrive in full (%d frames, %d bytes): the stream was cut or altered", frames, len(content))
	}
	if strings.Contains(warnings, responseJailbreakWarningCode) {
		return fmt.Errorf("a streamed response carried %s = %q; streamed responses are not scored, so the warning cannot come from a detection", responseWarningsHeader, warnings)
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

// responseJailbreakStreamedContent joins the content deltas of an OpenAI chat
// completion SSE stream. mock-vllm streams the echo in fixed-size chunks, so
// the phrase under test can straddle two frames and has to be reassembled.
func responseJailbreakStreamedContent(streamBody []byte) (string, int) {
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
