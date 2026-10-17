/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testcases

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	handoffRequestHeader = "x-vsr-handoff-envelope"
	handoffVersionHeader = "x-vsr-handoff-version"
	handoffIDHeader      = "x-vsr-handoff-id"
	handoffStatusHeader  = "x-vsr-handoff-status"
	handoffReasonHeader  = "x-vsr-handoff-reason"
	selectedModelHeader  = "x-vsr-selected-model"
	handoffE2EPrompt     = "What is the derivative of f(x) = x^3?"
)

type handoffE2EEnvelope struct {
	Version            string         `json:"version"`
	HandoffID          string         `json:"handoff_id"`
	RootInvocationID   string         `json:"root_invocation_id"`
	ParentInvocationID string         `json:"parent_invocation_id,omitempty"`
	State              string         `json:"state,omitempty"`
	ExpiresAt          string         `json:"expires_at"`
	Selection          map[string]any `json:"selection,omitempty"`
	Runtime            map[string]any `json:"runtime,omitempty"`
}

type handoffHTTPResult struct {
	status int
	header http.Header
	body   []byte
}

type handoffE2EClient struct {
	http        *http.Client
	gatewayURL  string
	providerURL string
}

func init() {
	pkgtestcases.Register("agentgateway-handoff-envelope", pkgtestcases.TestCase{
		Description: "Verify trusted Chat Completions handoff envelopes return receipts, are idempotent on retry, honor cancellation and expiry, leave routing unchanged, and stay out of provider requests",
		Tags:        []string{"agentgateway", "gateway", "routing", "handoff"},
		Fn:          testAgentGatewayHandoffEnvelope,
	})
}

func testAgentGatewayHandoffEnvelope(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	const (
		gatewayLocalPort  = "18083"
		providerLocalPort = "18084"
	)
	stopGateway, err := helpers.StartPortForward(
		ctx, client, opts.RestConfig, "agentgateway-system", "agentgateway-proxy",
		gatewayLocalPort+":80", opts.Verbose,
	)
	if err != nil {
		return fmt.Errorf("start agentgateway port-forward: %w", err)
	}
	defer stopGateway()
	stopProvider, err := helpers.StartPortForward(
		ctx, client, opts.RestConfig, "default", "vllm-llama3-8b-instruct",
		providerLocalPort+":8000", opts.Verbose,
	)
	if err != nil {
		return fmt.Errorf("start mock provider port-forward: %w", err)
	}
	defer stopProvider()
	time.Sleep(2 * time.Second)

	e2e := handoffE2EClient{
		http:        &http.Client{Timeout: 30 * time.Second},
		gatewayURL:  "http://localhost:" + gatewayLocalPort + "/v1/chat/completions",
		providerURL: "http://localhost:" + providerLocalPort + "/debug/last-request",
	}
	// The Router ledger outlives a single run, so IDs are unique per run.
	runID := fmt.Sprintf("%d", time.Now().UnixNano())

	baselineModel, err := e2e.checkRoutingIsUnchanged(ctx, runID)
	if err != nil {
		return err
	}
	if err := e2e.checkRetryAndConflict(ctx, runID); err != nil {
		return err
	}
	if err := e2e.checkCancellation(ctx, runID); err != nil {
		return err
	}
	if err := e2e.checkRejections(ctx, runID); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"selected_model": baselineModel})
	}
	return nil
}

// checkRoutingIsUnchanged sends the same prompt with and without an envelope.
// Version 1 admits the envelope without consuming selection facts, so the
// Router must select the same model and the provider must see no envelope data.
func (c handoffE2EClient) checkRoutingIsUnchanged(ctx context.Context, runID string) (string, error) {
	const opaqueRef = "opaque-ref-must-not-reach-model"
	const taskSummary = "task-summary-must-not-reach-model"

	baseline, err := c.send(ctx, "handoff-baseline-"+runID, nil)
	if err != nil {
		return "", err
	}
	if err := requireHandoffResult(baseline, http.StatusOK, "", "", ""); err != nil {
		return "", fmt.Errorf("request without envelope: %w", err)
	}
	baselineModel := baseline.header.Get(selectedModelHeader)
	if baselineModel == "" {
		return "", fmt.Errorf("request without envelope omitted %s", selectedModelHeader)
	}

	envelope := newHandoffE2EEnvelope("handoff-partial-" + runID)
	envelope.ParentInvocationID = "parent-" + runID
	envelope.Selection = map[string]any{"delegated_role": "researcher", "required_capabilities": []string{"tools"}}
	envelope.Runtime = map[string]any{"task_summary": taskSummary, "tool_state_refs": []string{opaqueRef}}
	providerSession := "handoff-provider-" + runID
	result, err := c.send(ctx, providerSession, &envelope)
	if err != nil {
		return "", err
	}
	if err := requireHandoffResult(result, http.StatusOK, envelope.HandoffID, "partial", "selection_unconsumed"); err != nil {
		return "", fmt.Errorf("partial handoff: %w", err)
	}
	if model := result.header.Get(selectedModelHeader); model != baselineModel {
		return "", fmt.Errorf("handoff changed selection to %q, want %q", model, baselineModel)
	}
	if err := c.requireProviderConfidentiality(ctx, providerSession, opaqueRef, taskSummary); err != nil {
		return "", err
	}
	return baselineModel, nil
}

func (c handoffE2EClient) checkRetryAndConflict(ctx context.Context, runID string) error {
	envelope := newHandoffE2EEnvelope("handoff-retry-" + runID)
	first, err := c.send(ctx, "handoff-retry-first-"+runID, &envelope)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(first, http.StatusOK, envelope.HandoffID, "accepted", "admitted"); err != nil {
		return fmt.Errorf("first handoff: %w", err)
	}
	retry, err := c.send(ctx, "handoff-retry-second-"+runID, &envelope)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(retry, http.StatusOK, envelope.HandoffID, "duplicate", "idempotent_retry"); err != nil {
		return fmt.Errorf("retried handoff: %w", err)
	}

	reused := envelope
	reused.RootInvocationID = "root-other-" + runID
	conflict, err := c.send(ctx, "handoff-retry-conflict-"+runID, &reused)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(conflict, http.StatusConflict, envelope.HandoffID, "rejected", "handoff_id_conflict"); err != nil {
		return fmt.Errorf("reused handoff id: %w", err)
	}
	return nil
}

func (c handoffE2EClient) checkCancellation(ctx context.Context, runID string) error {
	envelope := newHandoffE2EEnvelope("handoff-cancel-" + runID)
	if _, err := c.send(ctx, "handoff-cancel-active-"+runID, &envelope); err != nil {
		return err
	}
	cancel := envelope
	cancel.State = "cancelled"
	cancelled, err := c.send(ctx, "handoff-cancel-record-"+runID, &cancel)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(cancelled, http.StatusConflict, envelope.HandoffID, "cancelled", "cancel_recorded"); err != nil {
		return fmt.Errorf("cancellation: %w", err)
	}
	retry, err := c.send(ctx, "handoff-cancel-retry-"+runID, &envelope)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(retry, http.StatusConflict, envelope.HandoffID, "cancelled", "handoff_cancelled"); err != nil {
		return fmt.Errorf("retry after cancellation: %w", err)
	}
	return nil
}

func (c handoffE2EClient) checkRejections(ctx context.Context, runID string) error {
	expired := newHandoffE2EEnvelope("handoff-expired-" + runID)
	expired.ExpiresAt = time.Now().UTC().Add(-time.Minute).Format(time.RFC3339)
	result, err := c.send(ctx, "handoff-expired-"+runID, &expired)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(result, http.StatusUnprocessableEntity, "", "expired", "expired"); err != nil {
		return fmt.Errorf("expired handoff: %w", err)
	}

	newer := newHandoffE2EEnvelope("handoff-newer-" + runID)
	newer.Version = "2"
	result, err = c.send(ctx, "handoff-newer-"+runID, &newer)
	if err != nil {
		return err
	}
	if err := requireHandoffResult(result, http.StatusBadRequest, "", "rejected", "unsupported_version"); err != nil {
		return fmt.Errorf("newer handoff version: %w", err)
	}
	return nil
}

func newHandoffE2EEnvelope(id string) handoffE2EEnvelope {
	return handoffE2EEnvelope{
		Version:          "1",
		HandoffID:        id,
		RootInvocationID: "root-agentgateway-test",
		ExpiresAt:        time.Now().UTC().Add(10 * time.Minute).Format(time.RFC3339),
	}
}

func (c handoffE2EClient) send(
	ctx context.Context,
	providerSession string,
	envelope *handoffE2EEnvelope,
) (handoffHTTPResult, error) {
	payload := fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":%q}],"max_tokens":64,"temperature":0}`, handoffE2EPrompt)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.gatewayURL, strings.NewReader(payload))
	if err != nil {
		return handoffHTTPResult{}, fmt.Errorf("create handoff request: %w", err)
	}
	request.Header.Set("content-type", "application/json")
	request.Header.Set("x-vsr-test-session-id", providerSession)
	if envelope != nil {
		envelopeJSON, err := json.Marshal(envelope)
		if err != nil {
			return handoffHTTPResult{}, fmt.Errorf("marshal handoff envelope: %w", err)
		}
		request.Header.Set(handoffRequestHeader, base64.RawURLEncoding.EncodeToString(envelopeJSON))
	}

	response, err := c.http.Do(request)
	if err != nil {
		return handoffHTTPResult{}, fmt.Errorf("send handoff request: %w", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return handoffHTTPResult{}, fmt.Errorf("read handoff response: %w", err)
	}
	return handoffHTTPResult{status: response.StatusCode, header: response.Header.Clone(), body: body}, nil
}

// requireHandoffResult checks the HTTP status and receipt. An empty id means
// the envelope never parsed, so no version or ID may be echoed.
func requireHandoffResult(result handoffHTTPResult, status int, id, handoffStatus, reason string) error {
	if result.status != status {
		return fmt.Errorf("HTTP status %d, want %d: %s", result.status, status, result.body)
	}
	wantVersion := ""
	if id != "" {
		wantVersion = "1"
	}
	for header, want := range map[string]string{
		handoffVersionHeader: wantVersion,
		handoffIDHeader:      id,
		handoffStatusHeader:  handoffStatus,
		handoffReasonHeader:  reason,
	} {
		if got := result.header.Get(header); got != want {
			return fmt.Errorf("%s=%q, want %q", header, got, want)
		}
	}
	return nil
}

func (c handoffE2EClient) requireProviderConfidentiality(
	ctx context.Context,
	providerSession string,
	forbidden ...string,
) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.providerURL, nil)
	if err != nil {
		return fmt.Errorf("create provider observation request: %w", err)
	}
	request.Header.Set("x-vsr-test-session-id", providerSession)
	response, err := c.http.Do(request)
	if err != nil {
		return fmt.Errorf("fetch provider observation: %w", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("read provider observation: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("provider observation status %d: %s", response.StatusCode, body)
	}
	var observed struct {
		Headers map[string]string `json:"headers"`
	}
	if err := json.Unmarshal(body, &observed); err != nil {
		return fmt.Errorf("decode provider observation: %w", err)
	}
	for key := range observed.Headers {
		if strings.EqualFold(key, handoffRequestHeader) {
			return fmt.Errorf("provider received the handoff carrier header")
		}
	}
	for _, value := range forbidden {
		if strings.Contains(string(body), value) {
			return fmt.Errorf("provider received handoff runtime state")
		}
	}
	return nil
}
