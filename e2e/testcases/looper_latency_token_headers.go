package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("looper-latency-token-headers", pkgtestcases.TestCase{
		Description: "Verify looper aggregate latency and token usage surface as x-vsr-looper-latency-ms and x-vsr-looper-*-tokens response headers (issue #2694)",
		Tags:        []string{"kubernetes", "routing", "looper", "headers"},
		Fn:          testLooperLatencyTokenHeaders,
	})
}

// looperMetricsProbeKeyword is the sentinel token wired into the kubernetes
// profile's looper_metrics_probe_decision (see
// e2e/profiles/ai-gateway/values.yaml). A prompt containing it deterministically
// routes to that confidence-algorithm decision, which executes the looper
// across two model refs.
const looperMetricsProbeKeyword = "__MULTI_MODEL_PROBE__"

// testLooperLatencyTokenHeaders asserts the response-header half of the #2694
// contract end-to-end: after a looper execution, the aggregate wall-clock
// latency and aggregate token usage (already recorded internally by #2593)
// are surfaced as x-vsr-debug-gated headers.
func testLooperLatencyTokenHeaders(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing looper latency/token usage response headers")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	query := "Diagnostics request " + looperMetricsProbeKeyword + " please run."

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", query, 30*time.Second)
	if err != nil {
		return fmt.Errorf("looper metrics request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-latency-token-headers", "Query: "+query)
		return fmt.Errorf("looper metrics request: %s", formatUnexpectedChatCompletionStatus(response))
	}

	// Guard: confirm the keyword rule actually matched the looper metrics probe
	// decision. Without this, a silently non-matching rule would make the
	// header assertions vacuous (all absent, but for the wrong reason).
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != "looper_metrics_probe_decision" {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-latency-token-headers", "Query: "+query)
		return fmt.Errorf("expected x-vsr-selected-decision=looper_metrics_probe_decision, got %q", decision)
	}

	// The floor is a strict "> 0", not merely "parses as an integer": the
	// decision's two model refs guarantee a real multi-call looper execution
	// against the base-model simulator, so a genuine measurement can never be
	// exactly zero. A ">= 0" floor would let a regression that always emits
	// "0" (aggregation silently broken, or ExecuteWithLatency wired to the
	// wrong response) pass this test vacuously.
	values, err := parsePositiveIntHeaders(response.Headers, opts.Verbose,
		"x-vsr-looper-latency-ms",
		"x-vsr-looper-prompt-tokens",
		"x-vsr-looper-completion-tokens",
		"x-vsr-looper-total-tokens",
	)
	if err != nil {
		return err
	}

	// The total must equal the sum of its parts; this catches an aggregation
	// bug that would otherwise slip through the per-header positivity checks.
	if got, want := values["x-vsr-looper-total-tokens"], values["x-vsr-looper-prompt-tokens"]+values["x-vsr-looper-completion-tokens"]; got != want {
		return fmt.Errorf("x-vsr-looper-total-tokens = %d, want prompt+completion = %d", got, want)
	}

	if opts.SetDetails != nil {
		details := map[string]interface{}{"decision": "looper_metrics_probe_decision"}
		for header, value := range values {
			details[header] = value
		}
		opts.SetDetails(details)
	}

	if opts.Verbose {
		fmt.Println("[Test] Looper latency/token usage response headers verified ✓")
	}
	return nil
}

// parsePositiveIntHeaders parses each named header as a strictly positive
// int64, returning an error naming the first header that fails to parse or
// isn't positive.
func parsePositiveIntHeaders(respHeaders http.Header, verbose bool, headerNames ...string) (map[string]int64, error) {
	values := make(map[string]int64, len(headerNames))
	for _, header := range headerNames {
		raw := respHeaders.Get(header)
		value, parseErr := strconv.ParseInt(raw, 10, 64)
		if parseErr != nil || value <= 0 {
			return nil, fmt.Errorf("header %s: expected a positive integer, got %q", header, raw)
		}
		values[header] = value
		if verbose {
			fmt.Printf("[Test]   %s = %d\n", header, value)
		}
	}
	return values, nil
}
