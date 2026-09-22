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
	pkgtestcases.Register("streaming-usage-settlement", pkgtestcases.TestCase{
		Description: "Verify streaming Chat dispatch always requests the usage chunk, settlement records it, and the client only sees usage it asked for (issue #3182)",
		Tags:        []string{"kubernetes", "streaming", "observability", "llm"},
		Fn:          testStreamingUsageSettlement,
	})
}

// usageSettlementModel is pinned by name. In this profile the public model
// name and the upstream model id are the same string (no external_model_ids
// mapping, so config.ResolveExternalModelID returns the name unchanged), the
// stream flag already matches, and a benign prompt triggers no mutating
// plugin. That is the byte-replay-eligible request shape from issue #3331:
// before the fix, the replayed client bytes reached the backend without the
// forced stream_options.include_usage, the backend emitted no usage chunk,
// and the router silently recorded no usage for the request.
const usageSettlementModel = "base-model"

// usageSettlementMetricPrompt and usageSettlementMetricCompletion are the
// model-scoped counters that stayed empty in #3331
// (src/semantic-router/pkg/observability/metrics/metrics.go).
const (
	usageSettlementMetricPrompt     = "llm_model_prompt_tokens_total"
	usageSettlementMetricCompletion = "llm_model_completion_tokens_total"
)

// streamingUsageProbe is one streaming sub-case. clientOptIn distinguishes
// the client that never mentions stream_options from the client that asks
// for usage itself.
type streamingUsageProbe struct {
	name        string
	prompt      string
	clientOptIn bool
}

// testStreamingUsageSettlement pins the streaming usage contract from #3182,
// in the request shape that regressed in #3331.
//
// The dispatch-side assertion is the load-bearing one. The simulator appends
// the usage payload to every stream it produces regardless of stream_options
// (tools/test/services/provider-mocker/provider_mocker/chat_wire.py), so
// usage settlement here cannot distinguish a router that requested the chunk
// from one that got it for free. What the simulator does observe faithfully
// is the dispatch body it received: /debug/last-request answers with the
// exact request JSON, and a body whose stream_options.include_usage is
// missing means the byte-replay path is once again forwarding original
// client bytes without the forced flag. Real OpenAI-compatible backends omit
// the usage chunk for such a body, which is precisely the silent accounting
// loss #3331 reported.
//
// The two sub-cases together also pin the client-facing half: the router
// strips the usage it forced when the client never asked
// (src/semantic-router/pkg/protocolcodec/chat_usage_filter.go), and passes
// it through when the client opted in. Settlement is asserted through the
// model-scoped token counters, which must grow across the probes.
func testStreamingUsageSettlement(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing streaming usage dispatch, stripping, and settlement")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	// The same simulator observation seam the short-circuit contract uses
	// (e2e/testcases/plugin_short_circuit_dispatch.go).
	backend, err := fixtures.OpenServiceEndpointSession(
		ctx, client, opts,
		shortCircuitBackendNamespace,
		shortCircuitBackendService,
		shortCircuitBackendPort,
	)
	if err != nil {
		return fmt.Errorf("failed to reach the upstream simulator: %w", err)
	}
	defer backend.Close()

	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer metricsSession.Close()

	promptBefore, completionBefore, err := scrapeModelTokenCounters(ctx, metricsSession, usageSettlementModel)
	if err != nil {
		return err
	}

	// The run id scopes the simulator sessions. It stays out of the prompts:
	// distinct digit-free prompts per sub-case keep classifiers and any
	// response cache out of the loop, following the short-circuit precedent.
	runID := fmt.Sprintf("%d", time.Now().UnixNano())

	probes := []streamingUsageProbe{
		{
			// The #3331 shape: pinned model, stream on, no stream_options.
			name:        "forced",
			prompt:      "Describe how transfer RNA carries amino acids to the ribosome",
			clientOptIn: false,
		},
		{
			// The client asks for usage itself, so the router must not strip
			// it. This control also proves the simulator emits the usage
			// payload, which makes the absence asserted above evidence of
			// stripping rather than of a mute backend.
			name:        "opt-in",
			prompt:      "Summarize why chloroplasts capture light energy for a plant cell",
			clientOptIn: true,
		},
	}

	details := map[string]interface{}{}
	for _, probe := range probes {
		sessionID := fmt.Sprintf("usage-settlement-%s-%s", probe.name, runID)
		result, probeErr := runStreamingUsageProbe(ctx, session, backend, probe, sessionID, opts.Verbose)
		if probeErr != nil {
			return probeErr
		}
		details[probe.name] = result
	}

	promptAfter, completionAfter, err := awaitModelTokenCounterIncrease(ctx, metricsSession, usageSettlementModel, promptBefore, completionBefore)
	if err != nil {
		return err
	}
	details["metrics"] = map[string]interface{}{
		"prompt_tokens_delta":     promptAfter - promptBefore,
		"completion_tokens_delta": completionAfter - completionBefore,
	}

	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	if opts.Verbose {
		fmt.Println("[Test] Streaming usage settlement contract verified")
	}
	return nil
}

func runStreamingUsageProbe(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backend *fixtures.ServiceSession,
	probe streamingUsageProbe,
	sessionID string,
	verbose bool,
) (map[string]interface{}, error) {
	response, err := sendStreamingUsageRequest(ctx, session, probe, sessionID)
	if err != nil {
		return nil, fmt.Errorf("%s: request failed: %w", probe.name, err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: expected status 200, got %d: %s",
			probe.name, response.StatusCode, truncateString(string(response.Body), 500))
	}
	if contentType := response.Headers.Get("Content-Type"); !strings.Contains(contentType, "text/event-stream") {
		return nil, fmt.Errorf("%s: expected an SSE response, got Content-Type %q", probe.name, contentType)
	}

	content, usageFrames, err := decodeStreamingUsageFrames(probe.name, response.Body)
	if err != nil {
		return nil, err
	}
	if content == "" {
		return nil, fmt.Errorf("%s: the stream carried no delta content", probe.name)
	}

	if probe.clientOptIn {
		if len(usageFrames) != 1 {
			return nil, fmt.Errorf("%s: the client opted into usage, expected exactly one usage-bearing frame, got %d",
				probe.name, len(usageFrames))
		}
		usage := usageFrames[0]
		if usage.PromptTokens <= 0 || usage.CompletionTokens <= 0 {
			return nil, fmt.Errorf("%s: expected positive usage counts, got prompt=%d completion=%d",
				probe.name, usage.PromptTokens, usage.CompletionTokens)
		}
	} else if len(usageFrames) != 0 {
		// The client never asked for usage, so the flag the router forced on
		// the dispatch must not leak accounting evidence into the public
		// stream (chat_usage_filter.go removes it).
		return nil, fmt.Errorf("%s: the client did not request usage but the stream carried %d usage-bearing frame(s)",
			probe.name, len(usageFrames))
	}

	dispatchBody, err := lookupStreamingUsageDispatch(ctx, backend, sessionID)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", probe.name, err)
	}
	if err := assertStreamingUsageDispatchBody(probe, dispatchBody); err != nil {
		return nil, err
	}

	if verbose {
		fmt.Printf("[Test]   %s: usage frames=%d content bytes=%d\n", probe.name, len(usageFrames), len(content))
	}

	return map[string]interface{}{
		"usage_frames":       len(usageFrames),
		"content_bytes":      len(content),
		"session_identifier": sessionID,
	}, nil
}

// assertStreamingUsageDispatchBody checks what the backend actually received.
// The include_usage assertion is the regression gate for #3331; the model and
// message assertions establish that nothing else rewrote the request, so the
// forced flag was the only thing standing between the client bytes and
// byte-replay.
func assertStreamingUsageDispatchBody(probe streamingUsageProbe, body map[string]interface{}) error {
	if model, _ := body["model"].(string); model != usageSettlementModel {
		return fmt.Errorf("%s: the dispatch body names model %q, expected the pinned %q; a model rewrite means this probe is not exercising the replay-eligible path",
			probe.name, model, usageSettlementModel)
	}
	if stream, _ := body["stream"].(bool); !stream {
		return fmt.Errorf("%s: the dispatch body does not carry stream=true", probe.name)
	}
	messages, _ := body["messages"].([]interface{})
	if len(messages) != 1 {
		return fmt.Errorf("%s: expected the single client message on the dispatch body, got %d messages", probe.name, len(messages))
	}
	message, _ := messages[0].(map[string]interface{})
	if role, _ := message["role"].(string); role != "user" {
		return fmt.Errorf("%s: expected the dispatched message role to stay user, got %q", probe.name, role)
	}
	if content, _ := message["content"].(string); content != probe.prompt {
		return fmt.Errorf("%s: the dispatched message content differs from the client prompt: %s",
			probe.name, truncateString(fmt.Sprintf("%v", content), 300))
	}

	streamOptions, ok := body["stream_options"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("%s: the dispatch body carries no stream_options, so the byte-replay path is bypassing the forced usage request again (issue #3331); a real backend would emit no usage chunk and the request would settle without usage",
			probe.name)
	}
	if includeUsage, _ := streamOptions["include_usage"].(bool); !includeUsage {
		return fmt.Errorf("%s: the dispatch body carries stream_options but include_usage is not true: %v",
			probe.name, streamOptions)
	}
	return nil
}

// streamingUsageFrameUsage is the usage payload of one SSE frame.
type streamingUsageFrameUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// decodeStreamingUsageFrames walks the SSE data frames, concatenates the
// delta content, and collects every frame that carries a non-null usage
// object.
func decodeStreamingUsageFrames(name string, body []byte) (string, []streamingUsageFrameUsage, error) {
	frames := protocolSSEDataFrames(body)
	if len(frames) < 2 {
		return "", nil, fmt.Errorf("%s: expected a chunked SSE stream, got %d data frame(s): %s",
			name, len(frames), truncateString(string(body), 500))
	}
	if frames[len(frames)-1] != "[DONE]" {
		return "", nil, fmt.Errorf("%s: the stream did not terminate with [DONE]: %s",
			name, truncateString(frames[len(frames)-1], 300))
	}

	var content strings.Builder
	var usageFrames []streamingUsageFrameUsage
	for _, frame := range frames[:len(frames)-1] {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *streamingUsageFrameUsage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			return "", nil, fmt.Errorf("%s: stream frame is not valid JSON: %w (frame: %s)",
				name, err, truncateString(frame, 300))
		}
		for _, choice := range chunk.Choices {
			content.WriteString(choice.Delta.Content)
		}
		if chunk.Usage != nil {
			usageFrames = append(usageFrames, *chunk.Usage)
		}
	}
	return content.String(), usageFrames, nil
}

func sendStreamingUsageRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	probe streamingUsageProbe,
	sessionID string,
) (*localChatCompletionResponse, error) {
	requestBody := map[string]interface{}{
		"model":  usageSettlementModel,
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": probe.prompt},
		},
	}
	if probe.clientOptIn {
		requestBody["stream_options"] = map[string]interface{}{"include_usage": true}
	}
	payload, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, session.URL(localChatCompletionsPath), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")
	req.Header.Set(shortCircuitSessionHeader, sessionID)

	resp, err := session.HTTPClient(60 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &localChatCompletionResponse{StatusCode: resp.StatusCode, Headers: resp.Header, Body: body}, nil
}

// lookupStreamingUsageDispatch fetches the request body the simulator
// recorded for the session. Unlike the short-circuit lookup this returns the
// whole body: the contract here is about what rode on the dispatch, not
// whether one happened.
func lookupStreamingUsageDispatch(ctx context.Context, backend *fixtures.ServiceSession, sessionID string) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, backend.URL("/debug/last-request"), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set(shortCircuitSessionHeader, sessionID)

	resp, err := backend.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return nil, fmt.Errorf("simulator observation request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("the simulator never saw session %q, so the streamed request either never dispatched or dropped the %s header; the dispatch-body assertions cannot run",
			sessionID, shortCircuitSessionHeader)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected simulator observation status %d: %s",
			resp.StatusCode, truncateString(string(body), 300))
	}

	var observation struct {
		Body map[string]interface{} `json:"body"`
	}
	if err := json.Unmarshal(body, &observation); err != nil {
		return nil, fmt.Errorf("simulator observation is not valid JSON: %w (body: %s)",
			err, truncateString(string(body), 300))
	}
	if observation.Body == nil {
		return nil, fmt.Errorf("the simulator recorded the session without a body: %s", truncateString(string(body), 300))
	}
	return observation.Body, nil
}

// scrapeModelTokenCounters reads the model-scoped prompt and completion
// counters. A series that does not exist yet reads as zero.
func scrapeModelTokenCounters(ctx context.Context, metricsSession *fixtures.ServiceSession, model string) (float64, float64, error) {
	resp, err := fixtures.DoGETRequest(ctx, metricsSession.HTTPClient(15*time.Second), metricsSession.URL("/metrics"))
	if err != nil {
		return 0, 0, fmt.Errorf("fetch /metrics: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("/metrics: expected 200, got %d", resp.StatusCode)
	}
	body := string(resp.Body)
	prompt, err := promModelCounterValue(body, usageSettlementMetricPrompt, model)
	if err != nil {
		return 0, 0, err
	}
	completion, err := promModelCounterValue(body, usageSettlementMetricCompletion, model)
	if err != nil {
		return 0, 0, err
	}
	return prompt, completion, nil
}

// awaitModelTokenCounterIncrease rescrapes until both counters grew past
// their starting values. Settlement lands when the extproc observes the end
// of the stream, which can trail the client's final byte by a moment.
func awaitModelTokenCounterIncrease(
	ctx context.Context,
	metricsSession *fixtures.ServiceSession,
	model string,
	promptBefore, completionBefore float64,
) (float64, float64, error) {
	var prompt, completion float64
	var err error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		prompt, completion, err = scrapeModelTokenCounters(ctx, metricsSession, model)
		if err != nil {
			return 0, 0, err
		}
		if prompt > promptBefore && completion > completionBefore {
			return prompt, completion, nil
		}
	}
	return prompt, completion, fmt.Errorf(
		"usage settlement never reached the model counters: %s stayed at %v (was %v) and %s at %v (was %v) for model %q; streamed usage is being dropped on the accounting path",
		usageSettlementMetricPrompt, prompt, promptBefore,
		usageSettlementMetricCompletion, completion, completionBefore, model)
}

// promModelCounterValue extracts a single-label counter sample from the
// Prometheus text exposition.
func promModelCounterValue(metricsBody, metricName, model string) (float64, error) {
	needle := fmt.Sprintf("%s{model=%q}", metricName, model)
	for _, line := range strings.Split(metricsBody, "\n") {
		if !strings.HasPrefix(line, needle) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("unparsable metrics line: %q", line)
		}
		value, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil {
			return 0, fmt.Errorf("unparsable value in metrics line %q: %w", line, err)
		}
		return value, nil
	}
	return 0, nil
}
