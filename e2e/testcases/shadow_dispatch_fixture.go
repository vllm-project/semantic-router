package testcases

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	shadowPrimaryModel   = "openai/gpt-oss-20b"
	shadowPrimaryContent = "Hello from openai/gpt-oss-20b."
	shadowPrimaryCeiling = 2 * time.Second
	shadowBarrierTimeout = 5 * time.Second
)

type shadowDispatchRun struct {
	gateway, api, provider, metrics *fixtures.ServiceSession
	chat                            *fixtures.ChatCompletionsClient
	client                          *kubernetes.Clientset
	opts                            pkgtestcases.TestCaseOptions
	baseline                        []byte
	details                         map[string]interface{}
	primaries                       []map[string]interface{}
	records                         []*shadowReplayRecord
	providerStates                  []shadowProviderState
}

func openShadowDispatchRun(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*shadowDispatchRun, error) {
	if opts.Profile != "router-replay" {
		return nil, fmt.Errorf("shadow isolation requires the router-replay profile, got %q", opts.Profile)
	}
	if _, found, postgresErr := getPostgresPod(ctx, client); postgresErr != nil {
		return nil, postgresErr
	} else if !found {
		return nil, fmt.Errorf("shadow isolation requires the profile's Postgres deployment")
	}
	run := &shadowDispatchRun{
		client: client, opts: opts,
		details: map[string]interface{}{
			"primary_ceiling_ms": shadowPrimaryCeiling.Milliseconds(),
			"warmup_requests":    2,
			"expected_model":     shadowPrimaryModel,
			"expected_content":   shadowPrimaryContent,
		},
	}
	if opts.SetDetails != nil {
		opts.SetDetails(run.details)
	}
	ready := false
	defer func() {
		if !ready {
			run.Close()
		}
	}()
	var err error
	run.gateway, err = fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	run.api, err = fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	run.provider, err = fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		return nil, err
	}
	run.metrics, err = fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	run.chat = fixtures.NewChatCompletionsClient(run.gateway, shadowDispatchPollTimeout)
	if _, err = run.primary(ctx, "auto", shadowDispatchPollTimeout); err != nil {
		return nil, fmt.Errorf("cold primary warmup: %w", err)
	}
	if _, err = run.primary(ctx, "auto", shadowPrimaryCeiling); err != nil {
		return nil, fmt.Errorf("warmed primary baseline: %w", err)
	}
	ready = true
	return run, nil
}

func (r *shadowDispatchRun) Close() {
	for _, session := range []*fixtures.ServiceSession{r.metrics, r.provider, r.api, r.gateway} {
		session.Close()
	}
}

func (r *shadowDispatchRun) primary(ctx context.Context, model string, ceiling time.Duration) (string, error) {
	sessionID := "e2e_shadow_" + uuid.NewString()
	requestCtx, cancel := context.WithTimeout(ctx, ceiling)
	defer cancel()
	started := time.Now()
	response, err := r.chat.Create(requestCtx, fixtures.ChatCompletionsRequest{
		Model: model,
		User:  "e2e-shadow-user",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Please say hello."},
		},
	}, map[string]string{
		"x-authz-user-id": "e2e-shadow-user",
		"x-session-id":    sessionID,
	})
	elapsed := time.Since(started)
	evidence := map[string]interface{}{
		"entrypoint": model, "session_id": sessionID,
		"elapsed_ms": float64(elapsed) / float64(time.Millisecond),
		"ceiling_ms": ceiling.Milliseconds(),
	}
	r.primaries = append(r.primaries, evidence)
	r.details["primary_responses"] = r.primaries
	if err != nil {
		return "", fmt.Errorf("primary %s failed after %s (ceiling %s): %w", sessionID, elapsed, ceiling, err)
	}
	evidence["status"] = response.StatusCode
	evidence["body"] = string(response.Body)
	if r.opts.Verbose {
		fmt.Printf("[Test] primary session=%s entrypoint=%s status=%d elapsed=%s ceiling=%s body=%s\n",
			sessionID, model, response.StatusCode, elapsed, ceiling, response.Body)
	}
	if response.StatusCode != http.StatusOK || elapsed > ceiling {
		return "", fmt.Errorf("primary %s status=%d elapsed=%s, want 200 within %s: %s",
			sessionID, response.StatusCode, elapsed, ceiling, response.Body)
	}
	normalized, err := normalizeShadowPrimary(response.Body)
	if err != nil {
		return "", err
	}
	if r.baseline == nil {
		r.baseline = normalized
	} else if !bytes.Equal(normalized, r.baseline) {
		return "", fmt.Errorf("primary %s body differs from baseline (excluding created): got %s, want %s",
			sessionID, normalized, r.baseline)
	}
	return sessionID, nil
}

func normalizeShadowPrimary(body []byte) ([]byte, error) {
	var completion struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Created int64  `json:"created"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, fmt.Errorf("decode primary completion: %w", err)
	}
	if completion.Object != "chat.completion" || completion.Model != shadowPrimaryModel ||
		completion.Created <= 0 || len(completion.Choices) != 1 {
		return nil, fmt.Errorf("unexpected primary completion envelope: %s", body)
	}
	choice := completion.Choices[0]
	if choice.Index != 0 || choice.Message.Role != "assistant" ||
		choice.Message.Content != shadowPrimaryContent || choice.FinishReason != "stop" {
		return nil, fmt.Errorf("primary must return exactly %q with finish_reason=stop: %s", shadowPrimaryContent, body)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	delete(fields, "created")
	return json.Marshal(fields)
}

func (r *shadowDispatchRun) replay(ctx context.Context, sessionID string, outcomes int) (*shadowReplayRecord, error) {
	record, err := waitForShadowReplay(ctx, r.api, sessionID, outcomes)
	if err != nil {
		return nil, err
	}
	r.records = append(r.records, record)
	r.details["replay_records"] = r.records
	if r.opts.Verbose {
		body, marshalErr := json.Marshal(record)
		if marshalErr != nil {
			return nil, marshalErr
		}
		fmt.Printf("[Test] replay=%s\n", body)
	}
	if record.SessionID != sessionID || record.SelectedModel != shadowPrimaryModel {
		return nil, fmt.Errorf("replay %s session=%q model=%q, want session=%q model=%q",
			record.ID, record.SessionID, record.SelectedModel, sessionID, shadowPrimaryModel)
	}
	normalized, err := normalizeShadowPrimary([]byte(record.ResponseBody))
	if err != nil {
		return nil, fmt.Errorf("replay %s primary body: %w", record.ID, err)
	}
	if !bytes.Equal(normalized, r.baseline) {
		return nil, fmt.Errorf("replay %s primary body differs from the client baseline: %s", record.ID, normalized)
	}
	if _, found, postgresErr := getPostgresPod(ctx, r.client); postgresErr != nil {
		return nil, postgresErr
	} else if !found {
		return nil, fmt.Errorf("postgres disappeared before confirming replay %s", record.ID)
	}
	if err = assertPostgresReplayRecordStored(ctx, r.client, record.ID, r.opts); err != nil {
		return nil, err
	}
	return record, nil
}

func requireShadowOutcome(record *shadowReplayRecord, model, verdict, reason string) error {
	if len(record.Outcomes) != 1 {
		return fmt.Errorf("replay %s has %d shadow outcomes, want one", record.ID, len(record.Outcomes))
	}
	outcome := record.Outcomes[0]
	if outcome.Target != "model" || outcome.TargetRef != model || outcome.Verdict != verdict || outcome.Reason != reason {
		return fmt.Errorf("replay %s shadow target=%s/%s verdict=%s reason=%s, want model/%s %s/%s",
			record.ID, outcome.Target, outcome.TargetRef, outcome.Verdict, outcome.Reason, model, verdict, reason)
	}
	if outcome.Metadata["primary_model"] != shadowPrimaryModel ||
		outcome.Metadata["shadow_model"] != model || outcome.Metadata["shadow_request_id"] == "" ||
		outcome.Metadata["attempts"] != "1" {
		return fmt.Errorf("replay %s unexpected shadow identity or attempts: %v", record.ID, outcome.Metadata)
	}
	if _, captured := outcome.Metadata["response_excerpt"]; captured {
		return fmt.Errorf("replay %s captured shadow text with capture_response_body disabled", record.ID)
	}
	if verdict == "completed" {
		expectedHash := fmt.Sprintf("%x", sha256.Sum256([]byte("Hello from "+model+".")))
		if outcome.Metadata["status_code"] != "200" || outcome.Metadata["response_sha256"] != expectedHash {
			return fmt.Errorf("replay %s did not record the expected shadow response: %v", record.ID, outcome.Metadata)
		}
	} else {
		if outcome.Metadata["error"] == "" {
			return fmt.Errorf("replay %s failed shadow has no error detail", record.ID)
		}
		if _, present := outcome.Metadata["response_sha256"]; present {
			return fmt.Errorf("replay %s failed shadow has response provenance", record.ID)
		}
	}
	return nil
}

type shadowProviderState struct {
	Mode       string   `json:"mode"`
	Received   int      `json:"received"`
	Active     int      `json:"active"`
	Expired    int      `json:"expired"`
	RequestIDs []string `json:"request_ids"`
}

func (r *shadowDispatchRun) control(ctx context.Context, scenario, action, mode string) error {
	var payload interface{}
	if action == "reset" {
		payload = map[string]string{"mode": mode}
	}
	response, err := fixtures.DoPOSTRequest(ctx, r.provider.HTTPClient(shadowBarrierTimeout),
		r.provider.URL("/debug/shadow/"+scenario+"/"+action), payload)
	if err != nil {
		return fmt.Errorf("shadow fixture %s/%s: %w", scenario, action, err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("shadow fixture %s/%s status %d: %s", scenario, action, response.StatusCode, response.Body)
	}
	return nil
}

func (r *shadowDispatchRun) releaseOnExit(ctx context.Context, scenario string, result *error) {
	cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shadowBarrierTimeout)
	defer cancel()
	if err := r.control(cleanupCtx, scenario, "release", ""); err != nil {
		*result = errors.Join(*result, fmt.Errorf("release shadow fixture: %w", err))
	}
}

func (r *shadowDispatchRun) waitProvider(ctx context.Context, scenario string, received, active int) (*shadowProviderState, error) {
	var state shadowProviderState
	pollErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, shadowBarrierTimeout, true, func(pollCtx context.Context) (bool, error) {
		response, fetchErr := fixtures.DoGETRequest(pollCtx, r.provider.HTTPClient(shadowBarrierTimeout),
			r.provider.URL("/debug/shadow/"+scenario))
		if fetchErr != nil {
			return false, fetchErr
		}
		if response.StatusCode != http.StatusOK {
			return false, fmt.Errorf("shadow fixture status %d: %s", response.StatusCode, response.Body)
		}
		if decodeErr := response.DecodeJSON(&state); decodeErr != nil {
			return false, decodeErr
		}
		if state.Expired != 0 || state.Received > received {
			return false, fmt.Errorf("shadow fixture %s exceeded its barrier or request count: %+v", scenario, state)
		}
		return state.Received == received && state.Active == active, nil
	})
	r.providerStates = append(r.providerStates, state)
	r.details["provider_"+scenario] = r.providerStates
	if r.opts.Verbose {
		fmt.Printf("[Test] provider scenario=%s state=%+v\n", scenario, state)
	}
	if pollErr != nil {
		return nil, fmt.Errorf("shadow fixture %s want received=%d active=%d, got %+v: %w", scenario, received, active, state, pollErr)
	}
	return &state, nil
}

type shadowQueueMetrics struct {
	Inflight float64 `json:"inflight"`
	Queued   float64 `json:"queued"`
	Drops    float64 `json:"queue_full_drops"`
}

func (r *shadowDispatchRun) queueMetrics(ctx context.Context) (shadowQueueMetrics, error) {
	body, err := fetchMetrics(ctx, r.metrics)
	if err != nil {
		return shadowQueueMetrics{}, err
	}
	var result shadowQueueMetrics
	for _, line := range strings.Split(body, "\n") {
		if !strings.Contains(line, `decision="shadow_queue_decision"`) {
			continue
		}
		var target *float64
		switch {
		case strings.HasPrefix(line, "sr_shadow_dispatch_inflight{"):
			target = &result.Inflight
		case strings.HasPrefix(line, "sr_shadow_dispatch_queued{"):
			target = &result.Queued
		case strings.HasPrefix(line, "sr_shadow_dispatch_total{") &&
			strings.Contains(line, `reason="queue_full"`) && strings.Contains(line, `result="dropped"`):
			target = &result.Drops
		default:
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return result, fmt.Errorf("invalid shadow metric: %q", line)
		}
		*target, err = strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return result, fmt.Errorf("parse shadow metric %q: %w", line, err)
		}
	}
	return result, nil
}

func (r *shadowDispatchRun) waitQueue(ctx context.Context, stage string, want shadowQueueMetrics) error {
	var got shadowQueueMetrics
	pollErr := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, shadowBarrierTimeout, true, func(pollCtx context.Context) (bool, error) {
		var fetchErr error
		got, fetchErr = r.queueMetrics(pollCtx)
		return got == want, fetchErr
	})
	r.details["queue_"+stage] = got
	if r.opts.Verbose {
		fmt.Printf("[Test] queue stage=%s got=%+v want=%+v\n", stage, got, want)
	}
	if pollErr != nil {
		return fmt.Errorf("queue stage %s got %+v, want %+v: %w", stage, got, want, pollErr)
	}
	return nil
}
