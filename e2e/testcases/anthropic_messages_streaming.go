package testcases

import (
	"bufio"
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
	pkgtestcases.Register("anthropic-messages-streaming", pkgtestcases.TestCase{
		Description: "Verify streaming /v1/messages response emits the Anthropic SSE event sequence",
		Tags:        []string{"anthropic", "streaming", "sse"},
		Fn:          testAnthropicMessagesStreaming,
	})
}

// anthropicStreamingEvent captures the per-line shape of an SSE frame:
// the "event:" name and the parsed "data:" payload. Sufficient for
// sequence-and-presence assertions; the field set inside each event
// is documented per-event-type by the Anthropic Messages API.
type anthropicStreamingEvent struct {
	Name string
	Data map[string]interface{}
}

// requiredAnthropicEventSequence is the strict ordering the SSE emitter
// must produce for a single-content-block response. message_start opens,
// at least one content_block (start + delta(s) + stop) per block, then
// message_delta carrying the terminal stop_reason, then message_stop
// closes. The emitter is free to interleave ping events; those are
// keepalive and intentionally not in this sequence.
var requiredAnthropicEventSequence = []string{
	"message_start",
	"content_block_start",
	"content_block_delta",
	"content_block_stop",
	"message_delta",
	"message_stop",
}

// testAnthropicMessagesStreaming asserts the SSE event sequence on a
// streaming /v1/messages response: at least one message_start, at least
// one content_block triplet (start/delta/stop), one message_delta with a
// stop_reason, and exactly one message_stop. These are the framing
// invariants any Anthropic SDK relies on to know when a response has
// fully arrived; missing any of them stalls or breaks the client.
func testAnthropicMessagesStreaming(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Verifying streaming /v1/messages SSE event sequence")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	resp, err := sendAnthropicMessagesStreamingRequest(ctx, anthropicMessagesRequestBody{
		Model:     "MoM",
		MaxTokens: 64,
		Stream:    true,
		Messages: []anthropicMessage{
			{Role: "user", Content: "Count to three."},
		},
	}, localPort)
	if err != nil {
		return fmt.Errorf("anthropic streaming request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("expected status 200 for streaming /v1/messages, got %d: %s",
			resp.StatusCode, truncateString(string(body), 200))
	}

	events, parseErrors, err := consumeAnthropicSSE(resp)
	if err != nil {
		return fmt.Errorf("failed to consume SSE stream: %w", err)
	}

	counts := countAnthropicEvents(events)
	if opts.SetDetails != nil {
		details := map[string]interface{}{
			"total_events": len(events),
		}
		for name, c := range counts {
			details["event_"+name] = c
		}
		if len(parseErrors) > 0 {
			details["parse_errors"] = parseErrors
		}
		opts.SetDetails(details)
	}

	return assertAnthropicStreamingInvariants(events, counts)
}

// assertAnthropicStreamingInvariants validates the framing rules for an
// Anthropic SSE stream: every required event type appears at least once,
// the events arrive in the documented order, and exactly one terminal
// message_stop closes the stream.
func assertAnthropicStreamingInvariants(events []anthropicStreamingEvent, counts map[string]int) error {
	for _, required := range requiredAnthropicEventSequence {
		if counts[required] == 0 {
			return fmt.Errorf("missing required Anthropic event %q in stream (saw %v)", required, counts)
		}
	}
	if counts["message_stop"] != 1 {
		return fmt.Errorf("expected exactly one message_stop event, got %d", counts["message_stop"])
	}
	if err := assertEventOrder(events, requiredAnthropicEventSequence); err != nil {
		return err
	}
	return nil
}

// sendAnthropicMessagesStreamingRequest POSTs a streaming /v1/messages
// body and returns the raw response so the caller can read SSE frames.
func sendAnthropicMessagesStreamingRequest(
	ctx context.Context,
	body anthropicMessagesRequestBody,
	localPort string,
) (*http.Response, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/messages", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("anthropic-version", "2023-06-01")

	// Streaming responses can take longer than the 30 s non-streaming default.
	httpClient := &http.Client{Timeout: 60 * time.Second}
	return httpClient.Do(req)
}

// consumeAnthropicSSE reads an Anthropic-format SSE stream into a slice
// of {name, data} frames. Each Anthropic SSE frame is a pair of lines:
//
//	event: <name>
//	data: <json>
//
// separated by a blank line. The emitter may also send ping events
// (just an event line plus an empty-data line); those are recorded so
// keepalive can be observed but are not asserted on.
//
// parseErrors accumulates any json.Unmarshal failures on data: lines so
// callers can surface them via opts.SetDetails for CI diagnostics. A
// non-nil parse error does not abort consumption — the event is still
// recorded with a nil Data map so sequence assertions can continue.
func consumeAnthropicSSE(resp *http.Response) ([]anthropicStreamingEvent, []string, error) {
	var events []anthropicStreamingEvent
	var parseErrors []string
	scanner := bufio.NewScanner(resp.Body)
	// SSE frames can be larger than the default 64 KiB scanner buffer
	// for long content_block_delta lines.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var pending anthropicStreamingEvent
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			pending.Name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			pending.Data = nil
			if data != "" {
				if err := json.Unmarshal([]byte(data), &pending.Data); err != nil {
					// Record the error so CI logs show the malformed payload
					// rather than a cryptic downstream assertion failure.
					parseErrors = append(parseErrors, fmt.Sprintf("event %q: %v", pending.Name, err))
				}
			}
		case line == "":
			if pending.Name != "" {
				events = append(events, pending)
			}
			pending = anthropicStreamingEvent{}
		}
	}
	if pending.Name != "" {
		events = append(events, pending)
	}
	if err := scanner.Err(); err != nil {
		return events, parseErrors, fmt.Errorf("scanner: %w", err)
	}
	return events, parseErrors, nil
}

// countAnthropicEvents tallies how many times each event name appeared.
func countAnthropicEvents(events []anthropicStreamingEvent) map[string]int {
	counts := make(map[string]int, len(requiredAnthropicEventSequence))
	for _, ev := range events {
		counts[ev.Name]++
	}
	return counts
}

// assertEventOrder verifies that the events of interest appear in the
// expected order (ignoring interleaved events such as ping or repeated
// content_block_delta frames). The check walks the actual stream once
// and advances through the expected sequence whenever it finds the
// next expected name.
func assertEventOrder(events []anthropicStreamingEvent, expected []string) error {
	idx := 0
	for _, ev := range events {
		if idx >= len(expected) {
			break
		}
		if ev.Name == expected[idx] {
			idx++
		}
	}
	if idx < len(expected) {
		var seen []string
		for _, ev := range events {
			seen = append(seen, ev.Name)
		}
		return fmt.Errorf(
			"event sequence broken: expected %v in order, got %v (matched %d/%d)",
			expected, seen, idx, len(expected),
		)
	}
	return nil
}
