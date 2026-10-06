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

package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// belowQuorumFusionRequest builds a three-model panel that yields exactly one
// usable response against a quorum of two.
func belowQuorumFusionRequest(fusion *config.FusionAlgorithmConfig) *Request {
	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: fusion}
	return req
}

func belowQuorumFusionConfig(policy config.FusionQuorumFailurePolicy, target string) *config.FusionAlgorithmConfig {
	return &config.FusionAlgorithmConfig{
		Model:                  "judge",
		AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
		MaxConcurrent:          3,
		MinSuccessfulResponses: 2,
		OnError:                config.FusionOnErrorSkip,
		QuorumFailurePolicy:    policy,
		QuorumFallbackTarget:   target,
	}
}

func TestFusionQuorumFailureDefaultsToTypedFailure(t *testing.T) {
	var judgeCalls, fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "judge":
			judgeCalls.Add(1)
			return "unexpected judge response", http.StatusOK
		case "backup-model":
			fallbackCalls.Add(1)
			return "unexpected fallback response", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig("", ""))
	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion panel quorum not met")
	assert.Zero(t, judgeCalls.Load(), "judge must not run below quorum")
	assert.Zero(t, fallbackCalls.Load(), "unset policy must not call a fallback")

	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, string(config.FusionQuorumFailurePolicyFail), evidence.SelectedPolicy)
	assert.Equal(t, FusionQuorumFailed, evidence.Disposition)
	assert.Empty(t, evidence.FallbackTarget)
}

func TestFusionQuorumFallbackServesTargetResponse(t *testing.T) {
	var judgeCalls, fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "judge":
			judgeCalls.Add(1)
			return "unexpected judge response", http.StatusOK
		case "backup-model":
			fallbackCalls.Add(1)
			return "fallback answer", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.EqualValues(t, 1, fallbackCalls.Load(), "fallback must run exactly once")
	assert.Zero(t, judgeCalls.Load(), "judge must not deliberate over an under-strength panel")
}

func TestFusionQuorumFallbackFailureKeepsQuorumCause(t *testing.T) {
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "backup-model":
			return "fallback unavailable", http.StatusBadGateway
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion quorum fallback to \"backup-model\" failed")

	// The quorum failure must stay reachable so callers can still classify the
	// root cause rather than only seeing the recovery failure.
	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, 2, evidence.RequiredCount)
	assert.Equal(t, 1, evidence.UsableCount)
	assert.Equal(t, "backup-model", evidence.FallbackTarget)
	assert.Equal(t, FusionQuorumFallbackFailed, evidence.Disposition)
}

// An empty fallback response is not a recovery. Serving it would reintroduce the
// degradation the quorum contract exists to prevent.
func TestFusionQuorumFallbackRejectsUnusableTargetResponse(t *testing.T) {
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "backup-model":
			return "   ", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no usable assistant content")
}

// A cancelled caller already gave up; spending more budget on a fallback would
// be wrong even when the policy allows one.
func TestFusionQuorumCancellationSkipsFallback(t *testing.T) {
	var fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		if model == "backup-model" {
			fallbackCalls.Add(1)
		}
		return "cancelled", http.StatusBadGateway
	})
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(ctx, req)

	require.Error(t, err)
	assert.Zero(t, fallbackCalls.Load(), "cancelled requests must not spend fallback budget")
}

// A non-quorum execution failure must never be converted into a served
// response by the policy layer.
func TestFusionQuorumPolicyIgnoresNonQuorumErrors(t *testing.T) {
	looper := NewFusionLooper(&config.LooperConfig{Endpoint: "http://127.0.0.1:0"})
	cfg := fusionExecutionConfig{
		QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget: "backup-model",
	}
	cause := assert.AnError

	resp, err := looper.applyQuorumFailurePolicy(
		context.Background(), newFusionTestRequest(), cfg, fusionPanelOutcome{}, cause)

	assert.Nil(t, resp)
	assert.ErrorIs(t, err, cause)
}

// The fallback serves the client, so its usability rule differs from a panel
// member's in both directions.
func TestFusionFallbackResponseUsabilityDiffersFromPanel(t *testing.T) {
	assert.False(t, isUsableFusionFallbackResponse(nil))
	assert.False(t, isUsableFusionFallbackResponse(&ModelResponse{}))
	assert.True(t, isUsableFusionFallbackResponse(&ModelResponse{Content: "answer"}))

	// A tool-only reply is not usable for a panel member (the judge has nothing
	// to deliberate over) but is a complete answer from a final-serving fallback.
	toolOnly := &ModelResponse{HasToolCalls: true}
	assert.False(t, isUsableFusionPanelResponse(toolOnly))
	assert.True(t, isUsableFusionFallbackResponse(toolOnly))

	// The converse: reasoning alone is panel evidence the judge can work with,
	// but no outbound codec emits it, so serving it would return an empty answer.
	reasoningOnly := &ModelResponse{ReasoningContent: "reasoning"}
	assert.True(t, isUsableFusionPanelResponse(reasoningOnly))
	assert.False(t, isUsableFusionFallbackResponse(reasoningOnly))
}

// Cancellation is decided from the caller context, not from timeout-shaped
// transport errors.
func TestFusionQuorumFallbackDispositionClassification(t *testing.T) {
	live := context.Background()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	expired, cancelExpired := context.WithDeadline(
		context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()

	cases := []struct {
		name string
		ctx  context.Context
		err  error
		want FusionQuorumDisposition
	}{
		{
			name: "stage window refusal",
			ctx:  live,
			err:  &StageContextWindowError{Model: "m", EstimatedTokens: 10, ContextWindow: 5},
			want: FusionQuorumBudgetExhausted,
		},
		{name: "caller cancelled", ctx: cancelled, err: context.Canceled, want: FusionQuorumCancelled},
		{name: "caller deadline elapsed", ctx: expired, err: context.DeadlineExceeded, want: FusionQuorumCancelled},
		{name: "caller cancelled during a transport failure", ctx: cancelled, err: assert.AnError, want: FusionQuorumCancelled},
		{name: "transport deadline under a live caller", ctx: live, err: context.DeadlineExceeded, want: FusionQuorumFallbackFailed},
		{name: "wrapped transport deadline under a live caller", ctx: live, err: fmt.Errorf("wrapped: %w", context.DeadlineExceeded), want: FusionQuorumFallbackFailed},
		{name: "ordinary failure", ctx: live, err: assert.AnError, want: FusionQuorumFallbackFailed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, fusionQuorumFallbackDisposition(tc.ctx, tc.err))
		})
	}
}

// The same distinction over the connector transport ExtProc actually uses.
//
// The connector applies its own AttemptTimeout on a derived context, so the
// resulting error looks like a deadline while the caller is still waiting. The
// unit cases cannot show that the caller context reaches the classifier.
func TestFusionQuorumFallbackTimeoutUnderLiveCallerIsFallbackFailure(t *testing.T) {
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "backup-model":
			// Outlive the one-second attempt timeout. A bounded sleep rather
			// than a blocking channel, so server.Close cannot deadlock.
			time.Sleep(1500 * time.Millisecond)
			return "too late", http.StatusOK
		default:
			return "failed", http.StatusBadGateway
		}
	})
	defer server.Close()

	cfg := &config.LooperConfig{Endpoint: server.URL, TimeoutSeconds: 1}
	client, err := NewConnectorClient(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))

	// borrowClient, not ownClient: t.Cleanup already closes this client, and
	// ownClient would make the Looper close it a second time.
	_, execErr := newFusionLooper(cfg, borrowClient(client)).Execute(context.Background(), req)

	require.Error(t, execErr)
	evidence, ok := FusionQuorumEvidenceFromError(execErr)
	require.True(t, ok, "the below-quorum cause must survive")
	assert.Equal(t, FusionQuorumFallbackFailed, evidence.Disposition,
		"a target that timed out under a live caller is a fallback failure")
}

// A served fallback returns no error, so the outcome must travel on the
// response for Replay and metrics to see it.
func TestFusionQuorumFallbackSuccessCarriesOutcome(t *testing.T) {
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "backup-model":
			return "fallback answer", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp.QuorumOutcome)
	assert.Equal(t, 2, resp.QuorumOutcome.RequiredCount)
	assert.Equal(t, 1, resp.QuorumOutcome.UsableCount)
	assert.Equal(t, string(config.FusionQuorumFailurePolicyFallback), resp.QuorumOutcome.SelectedPolicy)
	assert.Equal(t, "backup-model", resp.QuorumOutcome.FallbackTarget)
	// The looper reports readiness, not a terminal outcome: protocol translation
	// still runs at the ExtProc boundary and can fail. ExtProc promotes this to
	// fallback_served only once encoding has succeeded.
	assert.Equal(t, FusionQuorumFallbackReady, resp.QuorumOutcome.Disposition)

	// ModelsUsed and Iterations must reflect every call, not just the fallback.
	assert.Equal(t, []string{"panel-a", "panel-b", "panel-c", "backup-model"}, resp.ModelsUsed)
	assert.Equal(t, 4, resp.Iterations)
}

// The fallback is the final-serving stage, so it must run as the call after the
// whole panel rather than reusing the first panel iteration.
func TestFusionFallbackUsesPostPanelIteration(t *testing.T) {
	var fallbackIteration atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		if payload.Model == "backup-model" {
			if raw := r.Header.Get("x-vsr-looper-iteration"); raw != "" {
				var parsed int64
				_, _ = fmt.Sscanf(raw, "%d", &parsed)
				fallbackIteration.Store(parsed)
			}
			writeFusionTestCompletion(w, payload.Model, "fallback answer", http.StatusOK)
			return
		}
		if payload.Model == "panel-a" {
			writeFusionTestCompletion(w, payload.Model, "panel a answer", http.StatusOK)
			return
		}
		writeFusionTestCompletion(w, payload.Model, "failed", http.StatusBadGateway)
	}))
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	assert.Equal(t, 4, resp.Iterations)
	assert.EqualValues(t, 4, fallbackIteration.Load(),
		"fallback must report the post-panel iteration, not iteration 1")
}

// A served fallback must still expose the panel failure classes, which are the
// only way an operator can tell why the panel missed quorum.
func TestFusionFallbackSuccessCarriesAttemptEvidence(t *testing.T) {
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "backup-model":
			return "fallback answer", http.StatusOK
		default:
			return "failed", http.StatusBadGateway
		}
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp.QuorumOutcome)
	require.Len(t, resp.QuorumOutcome.Attempts, 3)
	states := make([]FusionPanelAttemptState, 0, 3)
	for _, attempt := range resp.QuorumOutcome.Attempts {
		states = append(states, attempt.State)
	}
	assert.Equal(t, []FusionPanelAttemptState{
		FusionPanelAttemptUsable,
		FusionPanelAttemptFailed,
		FusionPanelAttemptFailed,
	}, states)
}

// Early quorum must not be delayed by a slow worker whose result is discarded,
// even when no round timeout is configured.
func TestFusionPanelQuorumDoesNotWaitForDiscardedWorker(t *testing.T) {
	server := newFusionStubServer(t, func(model string, prompt string) (string, int) {
		switch model {
		case "panel-a", "panel-b":
			return model + " answer", http.StatusOK
		case "panel-c":
			time.Sleep(3 * time.Second)
			return "too late", http.StatusOK
		default:
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["a"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
			}
			return "final", http.StatusOK
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
			MaxConcurrent:          3,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	start := time.Now()
	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	elapsed := time.Since(start)

	require.NoError(t, err)
	assert.Less(t, elapsed, 2*time.Second,
		"quorum must return without waiting for the discarded slow worker")
}

// on_error: fail takes precedence over the panel-level fallback.
//
// The two settings answer different questions -- one about a single attempt, one
// about the panel as a whole -- and on_error predates the quorum policy. A
// recipe that sets both must still get the established fail-fast behavior, so
// the abort must not be converted into a quorum outcome that spends a fallback
// call the recipe owner did not ask for on this path.
func TestFusionOnErrorFailTakesPrecedenceOverQuorumFallback(t *testing.T) {
	var judgeCalls, fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "panel upstream failure", http.StatusBadGateway
		case "judge":
			judgeCalls.Add(1)
			return "unexpected judge response", http.StatusOK
		case "backup-model":
			fallbackCalls.Add(1)
			return "fallback answer", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	cfg := belowQuorumFusionConfig(config.FusionQuorumFailurePolicyFallback, "backup-model")
	cfg.OnError = config.FusionOnErrorFail
	req := belowQuorumFusionRequest(cfg)

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
		Execute(context.Background(), req)

	require.Error(t, err)
	assert.Nil(t, resp, "a fail-fast abort must not produce a served response")
	assert.Contains(t, err.Error(), "fusion panel model",
		"the caller must receive the underlying attempt failure")

	// The abort is an ordinary execution error, not a below-quorum outcome. This
	// is what keeps the quorum policy from ever seeing it.
	_, isQuorum := FusionQuorumEvidenceFromError(err)
	assert.False(t, isQuorum,
		"a per-attempt fail-fast must not be classified as a quorum failure")

	assert.Zero(t, fallbackCalls.Load(),
		"on_error: fail must abort before the quorum fallback is considered")
	assert.Zero(t, judgeCalls.Load(),
		"an aborted panel must not reach judge synthesis")
}
