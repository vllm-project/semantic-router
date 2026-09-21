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
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The caller's budget and the panel's own round budget are separate contracts.
// These cases pin the boundary between them: an exhausted caller must stop all
// further spending, while an exhausted round must not, because the caller is
// still waiting for an answer.

// A caller whose deadline has already elapsed must not have budget spent on a
// fallback, even in the window before the context timer publishes Done. Reading
// ctx.Err() alone leaves that decision to timer delivery.
func TestApplyQuorumFailurePolicySkipsFallbackAfterElapsedParentDeadline(t *testing.T) {
	var fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model, _ string) (string, int) {
		if model == "fallback-target" {
			fallbackCalls.Add(1)
		}
		return "unexpected fallback", http.StatusOK
	})
	defer server.Close()

	cfg := fusionExecutionConfig{
		AnalysisModels:         []string{"panel-a", "panel-b"},
		MinSuccessfulResponses: 2,
		MaxConcurrent:          2,
		OnError:                config.FusionOnErrorSkip,
		QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
		QuorumFallbackTarget:   "fallback-target",
	}
	outcome := fusionPanelOutcome{
		attempts: []FusionPanelAttemptEvidence{{Model: "panel-a", State: FusionPanelAttemptFailed}},
	}
	panelErr := newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, context.DeadlineExceeded)

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	response, err := looper.applyQuorumFailurePolicy(
		newElapsedDeadlineContext(), newFusionTestRequest(), cfg, outcome, panelErr)

	require.Error(t, err)
	assert.Nil(t, response)
	assert.Zero(t, fallbackCalls.Load(), "fallback must not spend budget past the caller's deadline")

	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, FusionQuorumCancelled, evidence.Disposition)
}

// fusionCallerBudgetErr is the guard that decides whether any further budget may
// be spent, so each of its cases is pinned directly.
func TestFusionCallerBudgetErr(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	live, cancelLive := context.WithTimeout(context.Background(), time.Hour)
	defer cancelLive()

	tests := []struct {
		name string
		ctx  context.Context
		want error
	}{
		{name: "no deadline", ctx: context.Background(), want: nil},
		{name: "live deadline", ctx: live, want: nil},
		{name: "cancelled", ctx: cancelled, want: context.Canceled},
		// The window this guard exists for: deadline passed, timer not delivered.
		{name: "elapsed deadline not yet published", ctx: newElapsedDeadlineContext(), want: context.DeadlineExceeded},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, fusionCallerBudgetErr(test.ctx))
		})
	}
}

// An internal round timeout is not the caller's budget running out. The caller
// is still connected, so the fallback must still answer. This drives Execute so
// the whole dispatch path, not just the guard, is covered.
func TestFusionRoundTimeoutStillServesFallbackForLiveCaller(t *testing.T) {
	var judgeCalls, fallbackCalls atomic.Int64
	server := newFusionStubServer(t, func(model string, _ string) (string, int) {
		switch model {
		case "panel-slow":
			time.Sleep(3 * time.Second)
			return "too late", http.StatusOK
		case "panel-b":
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

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-slow", "panel-b"},
			MaxConcurrent:          2,
			MinSuccessfulResponses: 2,
			RoundTimeoutSeconds:    1,
			OnError:                config.FusionOnErrorSkip,
			QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:   "backup-model",
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
		Execute(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Contains(t, string(resp.Body), "fallback answer")
	assert.EqualValues(t, 1, fallbackCalls.Load(), "a live caller must still receive the fallback")
	assert.Zero(t, judgeCalls.Load(), "judge must not run on a below-quorum panel")
}
