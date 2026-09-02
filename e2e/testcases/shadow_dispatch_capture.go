package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// Shadow dispatch contract: the primary response is served normally and the
// replay record for that request gains exactly one shadow_dispatch outcome
// whose verdict reflects the shadow backend, never the other way around.

const (
	shadowDispatchOutcomeSource = "shadow_dispatch"
	shadowDispatchPollTimeout   = 45 * time.Second
	shadowDispatchPollInterval  = 2 * time.Second
)

func init() {
	pkgtestcases.Register("shadow-dispatch-observes-candidate-model", pkgtestcases.TestCase{
		Description: "A sampled shadow call to a healthy candidate model completes and is recorded on the primary replay record without changing the primary response",
		Tags:        []string{"router-replay", "shadow-dispatch", "functional"},
		Fn:          testShadowDispatchObservesCandidateModel,
	})
	pkgtestcases.Register("shadow-dispatch-fail-open-unreachable-backend", pkgtestcases.TestCase{
		Description: "An unreachable shadow backend leaves the primary response intact and records a failed shadow outcome with a deterministic reason",
		Tags:        []string{"router-replay", "shadow-dispatch", "failure-isolation"},
		Fn:          testShadowDispatchFailOpenUnreachableBackend,
	})
}

func testShadowDispatchObservesCandidateModel(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	outcome, err := runShadowDispatchScenario(ctx, client, opts, "vllm-sr/shadow-ok")
	if err != nil {
		return err
	}
	if outcome.Verdict != "completed" || outcome.Reason != "completed" {
		return fmt.Errorf("shadow outcome verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if outcome.TargetRef != "openai/shadow-candidate" {
		return fmt.Errorf("shadow outcome target_ref=%q, want openai/shadow-candidate", outcome.TargetRef)
	}
	for _, key := range []string{"shadow_request_id", "latency_ms", "response_sha256", "status_code"} {
		if strings.TrimSpace(outcome.Metadata[key]) == "" {
			return fmt.Errorf("shadow outcome metadata missing %q: %v", key, outcome.Metadata)
		}
	}
	if _, captured := outcome.Metadata["response_excerpt"]; captured {
		return fmt.Errorf("shadow outcome stored response text although capture_response_body is off")
	}
	return nil
}

func testShadowDispatchFailOpenUnreachableBackend(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	outcome, err := runShadowDispatchScenario(ctx, client, opts, "vllm-sr/shadow-down")
	if err != nil {
		return err
	}
	if outcome.Verdict != "failed" {
		return fmt.Errorf("shadow outcome verdict=%q, want failed", outcome.Verdict)
	}
	switch outcome.Reason {
	case "transport_error", "timeout":
	default:
		return fmt.Errorf("shadow outcome reason=%q, want transport_error or timeout", outcome.Reason)
	}
	if _, provenance := outcome.Metadata["response_sha256"]; provenance {
		return fmt.Errorf("failed shadow outcome must not carry response provenance")
	}
	return nil
}

// runShadowDispatchScenario sends one chat completion into the given
// entrypoint, asserts the primary response contract, and returns the single
// shadow_dispatch outcome attached to that request's replay record.
func runShadowDispatchScenario(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	entrypointModel string,
) (*shadowReplayOutcome, error) {
	if opts.Verbose {
		fmt.Printf("[Test] Shadow dispatch via entrypoint %s\n", entrypointModel)
	}
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, fmt.Errorf("open session: %w", err)
	}
	defer session.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return nil, fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()

	sessionID := fmt.Sprintf("e2e_shadow_%d", time.Now().UnixNano())
	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	started := time.Now()
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: entrypointModel,
		User:  "e2e-shadow-user",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "shadow dispatch e2e prompt " + sessionID},
		},
	}, map[string]string{
		"x-authz-user-id": "e2e-shadow-user",
		"x-session-id":    sessionID,
	})
	if err != nil {
		return nil, fmt.Errorf("chat completions: %w", err)
	}
	primaryLatency := time.Since(started)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("primary chat completions status %d: %s", resp.StatusCode, string(resp.Body))
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(resp.Body, &completion); err != nil {
		return nil, fmt.Errorf("decode primary completion: %w", err)
	}
	if len(completion.Choices) == 0 || strings.TrimSpace(completion.Choices[0].Message.Content) == "" {
		return nil, fmt.Errorf("primary completion has no assistant content: %s", string(resp.Body))
	}
	if opts.Verbose {
		fmt.Printf("[Test] primary completion served in %s\n", primaryLatency)
	}

	return waitForShadowOutcome(ctx, apiSession, sessionID, opts.Verbose)
}

type shadowReplayOutcome struct {
	Source    string            `json:"source"`
	Target    string            `json:"target"`
	TargetRef string            `json:"target_ref"`
	Verdict   string            `json:"verdict"`
	Reason    string            `json:"reason"`
	Metadata  map[string]string `json:"metadata"`
}

type shadowReplayRecord struct {
	ID       string                `json:"id"`
	Outcomes []shadowReplayOutcome `json:"outcomes"`
}

// waitForShadowOutcome polls the replay record for the session until exactly
// one shadow_dispatch outcome is present. The shadow runs in the background,
// so the record can legitimately lag the primary response by a few seconds.
func waitForShadowOutcome(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
	verbose bool,
) (*shadowReplayOutcome, error) {
	deadline := time.Now().Add(shadowDispatchPollTimeout)
	var lastErr error
	for time.Now().Before(deadline) {
		outcome, err := fetchShadowOutcome(ctx, apiSession, sessionID)
		if err == nil {
			if verbose {
				fmt.Printf("[Test] shadow outcome verdict=%s reason=%s\n", outcome.Verdict, outcome.Reason)
			}
			return outcome, nil
		}
		lastErr = err
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(shadowDispatchPollInterval):
		}
	}
	return nil, fmt.Errorf("shadow outcome for session %q not recorded within %s: %w", sessionID, shadowDispatchPollTimeout, lastErr)
}

func fetchShadowOutcome(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
) (*shadowReplayOutcome, error) {
	items, err := fetchReplayListForSession(apiSession, sessionID, 5)
	if err != nil {
		return nil, err
	}
	if len(items) != 1 {
		return nil, fmt.Errorf("expected one replay row for session %q, got %d", sessionID, len(items))
	}
	raw, err := doRouterReplayManagementGET(ctx, apiSession, "/v1/router_replay/"+items[0].ID)
	if err != nil {
		return nil, fmt.Errorf("GET replay record: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET replay record status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var record shadowReplayRecord
	if err := raw.DecodeJSON(&record); err != nil {
		return nil, fmt.Errorf("decode replay record: %w", err)
	}
	var found []shadowReplayOutcome
	for _, outcome := range record.Outcomes {
		if outcome.Source == shadowDispatchOutcomeSource {
			found = append(found, outcome)
		}
	}
	if len(found) != 1 {
		return nil, fmt.Errorf("replay record %s has %d shadow outcomes, want 1", record.ID, len(found))
	}
	return &found[0], nil
}
