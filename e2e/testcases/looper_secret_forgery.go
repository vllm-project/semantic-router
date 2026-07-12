package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	looperRequestHeader    = "x-vsr-looper-request"
	looperSecretHeader     = "x-vsr-looper-secret"
	looperDecisionHeader   = "x-vsr-looper-decision"
	looperIterationHeader  = "x-vsr-looper-iteration"
	looperAlgorithmHeader  = "x-vsr-looper-algorithm"
	looperIterationsHeader = "x-vsr-looper-iterations"
	looperModelsUsedHeader = "x-vsr-looper-models-used"
	responsePathHeader     = "x-vsr-response-path"
	selectedDecisionHeader = "x-vsr-selected-decision"
	fastResponseHeader     = "x-vsr-fast-response"
	forgedLooperDecision   = "block_pii"
	looperProbeDecision    = "looper_auth_probe_decision"
	looperProbeKeyword     = "__LOOPER_AUTH_PROBE__"
	// Valid 256-bit hex encoding exercises authentication failure rather than
	// only malformed-secret rejection.
	forgedLooperSecret = "0000000000000000000000000000000000000000000000000000000000000000"
)

func init() {
	pkgtestcases.Register("looper-secret-forgery", pkgtestcases.TestCase{
		Description: "Accept authenticated Looper reentry and reject forged internal headers",
		Tags:        []string{"kubernetes", "routing", "looper", "security", "headers"},
		Fn:          testLooperSecretForgery,
	})
}

type looperForgeryAttempt struct {
	name   string
	secret string
}

func testLooperSecretForgery(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	if err := verifyAuthenticatedLooperControl(ctx, localPort); err != nil {
		return err
	}

	attacks := []looperForgeryAttempt{
		{name: "missing secret"},
		{name: "wrong secret", secret: forgedLooperSecret},
	}
	for _, attack := range attacks {
		if err := verifyRejectedLooperForgery(ctx, localPort, attack); err != nil {
			return err
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"control_decision":        looperProbeDecision,
			"control_response_path":   "looper",
			"control_algorithm":       "ratings",
			"control_iterations":      2,
			"forgeries_rejected":      len(attacks),
			"forgery_status":          http.StatusForbidden,
			"forgery_response_path":   "error",
			"forged_decision_blocked": true,
		})
	}
	if opts.Verbose {
		fmt.Println("[Test] Authenticated Looper reentry accepted; marker forgeries rejected ✓")
	}
	return nil
}

func verifyAuthenticatedLooperControl(ctx context.Context, localPort string) error {
	control, err := sendLocalChatCompletion(
		ctx,
		localPort,
		"MoM",
		"Authenticated Looper control "+looperProbeKeyword,
		30*time.Second,
	)
	if err != nil {
		return fmt.Errorf("control request failed: %w", err)
	}
	if control.StatusCode != http.StatusOK {
		return fmt.Errorf(
			"control request: %s",
			formatUnexpectedChatCompletionStatus(control),
		)
	}
	if got := control.Headers.Get(selectedDecisionHeader); got != looperProbeDecision {
		return fmt.Errorf("control selected decision = %q, want %s", got, looperProbeDecision)
	}
	if got := control.Headers.Get(responsePathHeader); got != "looper" {
		return fmt.Errorf("control response path = %q, want looper", got)
	}
	if got := control.Headers.Get(looperAlgorithmHeader); got != "ratings" {
		return fmt.Errorf("control Looper algorithm = %q, want ratings", got)
	}
	if got := control.Headers.Get(looperIterationsHeader); got != "2" {
		return fmt.Errorf("control Looper iterations = %q, want 2", got)
	}
	modelsUsed := control.Headers.Get(looperModelsUsedHeader)
	if !strings.Contains(modelsUsed, "general-expert") ||
		!strings.Contains(modelsUsed, "math-expert") {
		return fmt.Errorf(
			"control Looper models = %q, want general-expert and math-expert",
			modelsUsed,
		)
	}
	if got := control.Headers.Get(looperSecretHeader); got != "" {
		return fmt.Errorf("control response exposed the internal Looper credential")
	}

	var payload struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(control.Body, &payload); err != nil {
		return fmt.Errorf("decode authenticated Looper response: %w", err)
	}
	if len(payload.Choices) != 2 {
		return fmt.Errorf("control Looper choices = %d, want 2", len(payload.Choices))
	}
	return nil
}

func verifyRejectedLooperForgery(
	ctx context.Context,
	localPort string,
	attempt looperForgeryAttempt,
) error {
	requestHeaders := make(http.Header)
	requestHeaders.Set(looperRequestHeader, "true")
	requestHeaders.Set(looperDecisionHeader, forgedLooperDecision)
	requestHeaders.Set(looperIterationHeader, "1")
	if attempt.secret != "" {
		requestHeaders.Set(looperSecretHeader, attempt.secret)
	}

	response, err := sendLocalChatCompletionWithHeaders(
		ctx,
		localPort,
		"MoM",
		"Benign Looper forgery probe",
		30*time.Second,
		requestHeaders,
	)
	if err != nil {
		return fmt.Errorf("%s request failed: %w", attempt.name, err)
	}
	if response.StatusCode != http.StatusForbidden {
		return fmt.Errorf(
			"%s status = %d, want 403; headers=%v body=%s",
			attempt.name,
			response.StatusCode,
			response.Headers,
			string(response.Body),
		)
	}
	if got := response.Headers.Get(responsePathHeader); got != "error" {
		return fmt.Errorf("%s response path = %q, want error", attempt.name, got)
	}
	if got := response.Headers.Get(fastResponseHeader); got != "" {
		return fmt.Errorf("%s exposed fast-response header %q", attempt.name, got)
	}
	if got := response.Headers.Get(selectedDecisionHeader); got == forgedLooperDecision {
		return fmt.Errorf("%s trusted forged decision %q", attempt.name, got)
	}
	body := string(response.Body)
	if strings.Contains(body, "personal information") {
		return fmt.Errorf("%s executed the forged fast-response plugin", attempt.name)
	}
	if attempt.secret != "" && strings.Contains(body, attempt.secret) {
		return fmt.Errorf("%s reflected the forged secret", attempt.name)
	}
	return nil
}
