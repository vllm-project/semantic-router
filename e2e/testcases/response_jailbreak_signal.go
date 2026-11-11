package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("response-jailbreak-streaming-passthrough", pkgtestcases.TestCase{
		Description: "Verify a streamed answer under a response_guard that only annotates streams through as it arrives, and is scored by the response-direction jailbreak rule once the stream ends",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak", "streaming"},
		Fn:          testResponseJailbreakStreamingPassthrough,
	})
	pkgtestcases.Register("response-jailbreak-streaming-block", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "Verify a response_guard block rule holds a streamed answer until it has read the whole answer, then refuses it with the stream's error event",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak", "streaming"},
		Fn:          testResponseJailbreakStreamingBlock,
	})
	pkgtestcases.Register("response-jailbreak-streaming-pass", pkgtestcases.TestCase{
		Description: "Verify a response_guard block rule releases a streamed answer it passes as the whole stream, terminal event included",
		Tags:        []string{"kubernetes", "security", "jailbreak", "response-jailbreak", "streaming"},
		Fn:          testResponseJailbreakStreamingPass,
	})
}

// responseJailbreakSignalRule is the response-direction rule the profile
// declares, and the series name its extraction is counted under.
const responseJailbreakSignalRule = "unsafe_completion"

// responseGuardBufferedMetric counts the streamed answers the Router held for
// a response_guard, by decision and outcome.
const responseGuardBufferedMetric = "llm_response_guard_buffered_total"

// responseJailbreakStream is one streamed answer as the client received it,
// with the counters the Router reported around it.
type responseJailbreakStream struct {
	status   int
	header   http.Header
	body     string
	content  string
	frames   int
	scored   float64
	buffered float64
}

// streamResponseJailbreak sends prompt as a streaming chat completion and
// reports what the client received, how often the response-direction rule was
// scored and how often decision's held streams ended with outcome meanwhile.
func streamResponseJailbreak(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	prompt, decision, outcome string,
) (*responseJailbreakStream, error) {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	defer stopPortForward()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return nil, fmt.Errorf("open metrics session: %w", err)
	}
	defer metricsSession.Close()

	scoredBefore, bufferedBefore, err := readResponseJailbreakCounters(ctx, metricsSession, decision, outcome)
	if err != nil {
		return nil, err
	}
	resp, err := sendResponseJailbreakStreamingRequest(ctx, localPort, prompt)
	if err != nil {
		return nil, fmt.Errorf("streaming request failed: %w", err)
	}
	defer resp.Body.Close()
	streamBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read stream: %w", err)
	}
	scoredAfter, bufferedAfter, err := readResponseJailbreakCounters(ctx, metricsSession, decision, outcome)
	if err != nil {
		return nil, err
	}

	content, frames := chatStreamedContent(streamBody)
	stream := &responseJailbreakStream{
		status: resp.StatusCode, header: resp.Header, body: string(streamBody), content: content, frames: frames,
		scored: scoredAfter - scoredBefore, buffered: bufferedAfter - bufferedBefore,
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":    stream.status,
			"frames":         stream.frames,
			"content_bytes":  len(stream.content),
			"warnings":       stream.header.Get(warningsHeader),
			"response_guard": traceValue(stream.header, "response_guard"),
			"scored":         stream.scored,
			"buffered":       stream.buffered,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] status=%d frames=%d content_bytes=%d guard=%q scored=+%v buffered=+%v\n",
			stream.status, stream.frames, len(stream.content), traceValue(stream.header, "response_guard"), stream.scored, stream.buffered)
	}
	if stream.status != 200 || !strings.Contains(stream.header.Get("Content-Type"), "text/event-stream") {
		return nil, fmt.Errorf("expected a 200 event stream, got %d %q: %s",
			stream.status, stream.header.Get("Content-Type"), truncateString(stream.body, 400))
	}
	return stream, nil
}

func readResponseJailbreakCounters(ctx context.Context, metricsSession *fixtures.ServiceSession, decision, outcome string) (float64, float64, error) {
	body, err := fetchMetrics(ctx, metricsSession)
	if err != nil {
		return 0, 0, fmt.Errorf("read the router's metrics: %w", err)
	}
	return signalExtractionCount(body, "jailbreak", responseJailbreakSignalRule), responseGuardBufferedCount(body, decision, outcome), nil
}

// responseGuardBufferedCount reads decision's held streams that ended with
// outcome out of the Prometheus text exposition; no series reads as zero.
func responseGuardBufferedCount(metricsText, decision, outcome string) float64 {
	outcomeLabel := fmt.Sprintf("outcome=%q", outcome)
	for _, line := range strings.Split(metricsText, "\n") {
		if !strings.HasPrefix(line, responseGuardBufferedMetric+"{") ||
			!strings.Contains(line, outcomeLabel) || !strings.Contains(line, decision) {
			continue
		}
		fields := strings.Fields(line)
		if value, err := strconv.ParseFloat(fields[len(fields)-1], 64); err == nil {
			return value
		}
	}
	return 0
}

// testResponseJailbreakStreamingPassthrough pins true streaming for a guard
// that only annotates: the answer reaches the client as it is generated, the
// response-direction rule scores it once the stream ends, and its warning
// code cannot reach x-vsr-warnings, which left with the first byte.
//
// The delivery alone would also pass with the answer never scored, so both
// halves are asserted together.
func testResponseJailbreakStreamingPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a streamed answer under an annotating guard streams through and is scored")
	}
	stream, err := streamResponseJailbreak(ctx, client, opts,
		responseJailbreakPrompt(responseJailbreakWarnProbe, responseJailbreakPhrase), "response_jailbreak_warn", "annotated")
	if err != nil {
		return err
	}
	if !strings.Contains(stream.content, responseJailbreakPhrase) {
		return fmt.Errorf("the streamed content did not arrive in full (%d frames, %d bytes): the stream was cut or altered", stream.frames, len(stream.content))
	}
	if stream.scored < 1 {
		return fmt.Errorf("the streamed answer was not scored by rule %q: %s did not grow", responseJailbreakSignalRule, signalExtractionMetric)
	}
	if stream.buffered != 0 || traceParam(stream.header, "response_guard", "buffered") != "" {
		return fmt.Errorf("an annotating guard held the stream (%s grew by %v), want true streaming", responseGuardBufferedMetric, stream.buffered)
	}
	if strings.Contains(stream.header.Get(warningsHeader), responseJailbreakWarningCode) {
		return fmt.Errorf("a live stream carried %s = %q; its headers left before the guard read the answer",
			warningsHeader, stream.header.Get(warningsHeader))
	}
	return nil
}

// testResponseJailbreakStreamingBlock sends the answer the buffered block
// case refuses, as a stream. The Router holds the stream until the guard has
// read the whole answer, so its jailbreak content at the very end is found
// before any byte leaves, and the client receives the refusal as the Chat
// stream's error event and none of the answer.
func testResponseJailbreakStreamingBlock(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a block rule refuses a streamed answer it reads in full")
	}
	stream, err := streamResponseJailbreak(ctx, client, opts,
		responseJailbreakPrompt(responseJailbreakBlockProbe, responseJailbreakPhrase), "response_jailbreak_block", "blocked")
	if err != nil {
		return err
	}
	if !strings.Contains(stream.body, `data: {"error":`) || !strings.Contains(stream.body, "jailbreak content detected") {
		return fmt.Errorf("the stream did not end in the guard's refusal: %s", truncateString(stream.body, 400))
	}
	if stream.content != "" || strings.Contains(stream.body, responseJailbreakFiller) {
		return fmt.Errorf("%d bytes of the refused answer reached the client", len(stream.content))
	}
	if guard := traceValue(stream.header, "response_guard"); guard != "blocked" || traceParam(stream.header, "response_guard", "buffered") != "?1" {
		return fmt.Errorf("trace response_guard = %q, want blocked on a buffered stream", guard)
	}
	if stream.scored < 1 || stream.buffered != 1 {
		return fmt.Errorf("scored +%v, %s{outcome=blocked} +%v; want the answer scored and one held stream blocked",
			stream.scored, responseGuardBufferedMetric, stream.buffered)
	}
	return nil
}

// testResponseJailbreakStreamingPass is the same long stream through the same
// block rule without the jailbreak line: the guard passes it, and the client
// receives the whole stream it held.
func testResponseJailbreakStreamingPass(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing that a block rule releases a streamed answer it passes")
	}
	const trailing = "That is the whole history of navigation."
	prompt := responseJailbreakPrompt(responseJailbreakBlockProbe, trailing)
	stream, err := streamResponseJailbreak(ctx, client, opts, prompt, "response_jailbreak_block", "passed")
	if err != nil {
		return err
	}
	if !strings.Contains(stream.content, prompt) {
		return fmt.Errorf("the released stream carries %d content bytes without the whole echoed prompt, want the whole answer", len(stream.content))
	}
	if !strings.HasSuffix(stream.body, "data: [DONE]\n\n") || strings.Contains(stream.body, `"error"`) {
		return fmt.Errorf("the released stream does not end with its terminal event: %s", truncateString(stream.body[max(0, len(stream.body)-400):], 400))
	}
	if guard := traceValue(stream.header, "response_guard"); guard != "passed" || traceParam(stream.header, "response_guard", "buffered") != "?1" {
		return fmt.Errorf("trace response_guard = %q, want passed on a buffered stream", guard)
	}
	if stream.scored < 1 || stream.buffered != 1 {
		return fmt.Errorf("scored +%v, %s{outcome=passed} +%v; want the answer scored and one held stream passed",
			stream.scored, responseGuardBufferedMetric, stream.buffered)
	}
	return nil
}

func sendResponseJailbreakStreamingRequest(ctx context.Context, localPort, prompt string) (*http.Response, error) {
	requestBody := map[string]interface{}{
		"model":  "vllm-sr/auto",
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
// stream and reports how many frames carried them. provider-mocker streams the answer
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
