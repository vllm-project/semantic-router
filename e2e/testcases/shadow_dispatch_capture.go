package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	shadowDispatchOutcomeSource = "shadow_dispatch"
	shadowDispatchPollTimeout   = 45 * time.Second
	shadowDispatchPollInterval  = 2 * time.Second
	// routerReplayDetailToken carries the operator role from the profile
	// values. The viewer role lacks replay.detail, and the API then redacts
	// outcome reason, target_ref, and metadata, which these cases assert on.
	//nolint:gosec // G101: fixed non-production credential for the e2e replay fixture.
	routerReplayDetailToken = "router-replay-e2e-operator-token"
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
	record, err := runShadowDispatchScenario(ctx, client, opts, "vllm-sr/shadow-ok")
	if err != nil {
		return err
	}
	if err = requireShadowOutcome(record, "openai/shadow-candidate", "completed", "completed"); err != nil {
		return err
	}
	outcome := record.Outcomes[0]
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
	record, err := runShadowDispatchScenario(ctx, client, opts, "vllm-sr/shadow-down")
	if err != nil {
		return err
	}
	outcome := record.Outcomes[0]
	switch outcome.Reason {
	case "transport_error", "timeout":
	default:
		return fmt.Errorf("shadow outcome reason=%q, want transport_error or timeout", outcome.Reason)
	}
	return requireShadowOutcome(record, "openai/shadow-unreachable", "failed", outcome.Reason)
}

func runShadowDispatchScenario(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	entrypointModel string,
) (*shadowReplayRecord, error) {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	defer run.Close()
	sessionID, err := run.primary(ctx, entrypointModel, shadowPrimaryCeiling)
	if err != nil {
		return nil, err
	}
	return run.replay(ctx, sessionID, 1)
}

type shadowReplayOutcome struct {
	Source          string            `json:"source"`
	Target          string            `json:"target"`
	TargetRef       string            `json:"target_ref"`
	Verdict         string            `json:"verdict"`
	Reason          string            `json:"reason"`
	Metadata        map[string]string `json:"metadata"`
	ContentRedacted bool              `json:"content_redacted"`
}

type shadowReplayRecord struct {
	ID             string                `json:"id"`
	SessionID      string                `json:"session_id"`
	SelectedModel  string                `json:"selected_model"`
	ResponseBody   string                `json:"response_body"`
	ResponseStatus int                   `json:"response_status"`
	Outcomes       []shadowReplayOutcome `json:"outcomes"`
}

func waitForShadowReplay(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
	outcomeCount int,
) (*shadowReplayRecord, error) {
	var record *shadowReplayRecord
	pollErr := wait.PollUntilContextTimeout(ctx, shadowDispatchPollInterval, shadowDispatchPollTimeout, true, func(pollCtx context.Context) (bool, error) {
		var fetchErr error
		record, fetchErr = fetchShadowReplay(pollCtx, apiSession, sessionID)
		if fetchErr != nil || record == nil {
			return false, fetchErr
		}
		if len(record.Outcomes) > outcomeCount || (record.ResponseStatus != 0 && record.ResponseStatus != http.StatusOK) {
			return false, fmt.Errorf("replay %s has status %d and %d shadow outcomes, want 200 and %d",
				record.ID, record.ResponseStatus, len(record.Outcomes), outcomeCount)
		}
		return len(record.Outcomes) == outcomeCount && record.ResponseStatus == http.StatusOK && record.ResponseBody != "", nil
	})
	if pollErr != nil {
		return nil, fmt.Errorf("shadow replay for session %q not ready: last=%+v: %w", sessionID, record, pollErr)
	}
	return record, nil
}

func fetchShadowReplay(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
) (*shadowReplayRecord, error) {
	items, err := fetchReplayListForSession(apiSession, sessionID, 5)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	if len(items) != 1 {
		return nil, fmt.Errorf("expected one replay row for session %q, got %d", sessionID, len(items))
	}
	raw, err := doRouterReplayManagementGETAs(ctx, apiSession, "/api/v1/observability/replays/"+items[0].ID, routerReplayDetailToken)
	if err != nil {
		return nil, fmt.Errorf("GET replay record: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET replay record status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var record shadowReplayRecord
	if err = raw.DecodeJSON(&record); err != nil {
		return nil, fmt.Errorf("decode replay record: %w", err)
	}
	var found []shadowReplayOutcome
	for _, outcome := range record.Outcomes {
		if outcome.Source == shadowDispatchOutcomeSource {
			found = append(found, outcome)
		}
	}
	for _, outcome := range found {
		if outcome.ContentRedacted {
			return nil, fmt.Errorf("replay record %s shadow outcome was redacted; the detail token lacks replay.detail", record.ID)
		}
	}
	record.Outcomes = found
	return &record, nil
}
