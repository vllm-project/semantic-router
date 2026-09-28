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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func drainTestCollector(models []string, quorum int) *fusionPanelCollector {
	cfg := fusionExecutionConfig{
		AnalysisModels:         models,
		MinSuccessfulResponses: quorum,
		OnError:                config.FusionOnErrorSkip,
	}
	return newFusionPanelCollector(cfg, func() {})
}

// drainTestArbiter builds an arbiter whose round boundary is the supplied
// cutoff and whose parent boundary is unset, which is the shape the executor
// produces for a configured round timeout under an open-ended caller context.
func drainTestArbiter(models []string, quorum int, cutoff time.Time) *fusionPanelArbiter {
	return newFusionPanelArbiter(
		fusionPanelBoundaries{roundDeadline: cutoff},
		drainTestCollector(models, quorum),
	)
}

// Results completed before the cutoff keep full semantics: they count toward
// quorum and their usage is retained.
func TestArbiterDrainQueuedPreservesQuorumSemantics(t *testing.T) {
	cutoff := time.Now()
	arbiter := drainTestArbiter([]string{"panel-a", "panel-b"}, 2, cutoff)
	results := make(chan fusionPanelResult, 2)
	results <- fusionPanelResult{index: 0, model: "panel-a", completedAt: cutoff.Add(-time.Second), resp: &ModelResponse{
		Content: "a", Usage: TokenUsage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
	}}
	results <- fusionPanelResult{index: 1, model: "panel-b", completedAt: cutoff.Add(-time.Second), resp: &ModelResponse{
		Content: "b", Usage: TokenUsage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
	}}

	require.True(t, arbiter.drainQueued(context.Background(), results),
		"two usable responses must satisfy a quorum of two")

	outcome, err := arbiter.decided()
	require.NoError(t, err)
	assert.Len(t, outcome.responses, 2)
	assert.Equal(t, TokenUsage{PromptTokens: 10, CompletionTokens: 10, TotalTokens: 20}, outcome.usage)
}

// Channel readiness is not proof of ordering: a worker can enqueue after the
// deadline is observed. Such a result must be accounted but must never satisfy
// quorum, otherwise the deadline outcome becomes nondeterministic.
func TestArbiterDrainQueuedRejectsLateArrivals(t *testing.T) {
	cutoff := time.Now()
	arbiter := drainTestArbiter([]string{"panel-a", "panel-b"}, 2, cutoff)
	results := make(chan fusionPanelResult, 2)
	results <- fusionPanelResult{index: 0, model: "panel-a", completedAt: cutoff.Add(time.Millisecond), resp: &ModelResponse{
		Content: "a", Usage: TokenUsage{TotalTokens: 10},
	}}
	results <- fusionPanelResult{index: 1, model: "panel-b", completedAt: cutoff.Add(time.Millisecond), resp: &ModelResponse{
		Content: "b", Usage: TokenUsage{TotalTokens: 10},
	}}

	assert.False(t, arbiter.drainQueued(context.Background(), results),
		"post-cutoff responses must not satisfy quorum")

	outcome, err := arbiter.decided()
	require.NoError(t, err)
	assert.Empty(t, outcome.responses)
	// Their usage is still real work that was paid for.
	assert.EqualValues(t, 20, outcome.usage.TotalTokens)
}

// Once a terminal decision is reached the drain must keep going, so results
// queued behind the deciding item are not dropped from evidence or usage.
func TestArbiterDrainQueuedKeepsDrainingAfterTerminalDecision(t *testing.T) {
	cutoff := time.Now()
	before := cutoff.Add(-time.Second)
	arbiter := drainTestArbiter([]string{"panel-a", "panel-b", "panel-c"}, 2, cutoff)
	results := make(chan fusionPanelResult, 3)
	results <- fusionPanelResult{index: 0, model: "panel-a", completedAt: before, resp: &ModelResponse{
		Content: "a", Usage: TokenUsage{TotalTokens: 10},
	}}
	results <- fusionPanelResult{index: 1, model: "panel-b", completedAt: before, resp: &ModelResponse{
		Content: "b", Usage: TokenUsage{TotalTokens: 10},
	}}
	results <- fusionPanelResult{index: 2, model: "panel-c", completedAt: before, resp: &ModelResponse{
		Content: "c", Usage: TokenUsage{TotalTokens: 7},
	}}

	require.True(t, arbiter.drainQueued(context.Background(), results))

	outcome, err := arbiter.decided()
	require.NoError(t, err)
	assert.EqualValues(t, 27, outcome.usage.TotalTokens,
		"every completed attempt must be accounted, including those behind the deciding result")
	require.Len(t, outcome.attempts, 3)
	assert.Equal(t, FusionPanelAttemptUsable, outcome.attempts[2].State)
}

// Results that land after the panel decision must keep their real state and
// usage, but must not retroactively satisfy quorum.
func TestDrainUncountedResultsAccountsWithoutSatisfyingQuorum(t *testing.T) {
	collector := drainTestCollector([]string{"panel-a", "panel-b"}, 2)
	results := make(chan fusionPanelResult, 2)
	results <- fusionPanelResult{index: 0, model: "panel-a", resp: &ModelResponse{
		Content: "a", Usage: TokenUsage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
	}}
	results <- fusionPanelResult{index: 1, model: "panel-b", resp: &ModelResponse{
		Content: "b", Usage: TokenUsage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
	}}

	collector.drainUncountedResults(results)
	outcome := collector.outcome()

	// Usage is real work that was paid for and must survive.
	assert.Equal(t, TokenUsage{PromptTokens: 10, CompletionTokens: 10, TotalTokens: 20}, outcome.usage)
	// Each state is the one the result carried.
	require.Len(t, outcome.attempts, 2)
	for _, attempt := range outcome.attempts {
		assert.Equal(t, FusionPanelAttemptUsable, attempt.State)
	}
	// But they do not count toward quorum.
	assert.Empty(t, outcome.responses)
}

// Without the drain, completed work was discarded and overwritten with
// synthetic cancelled states. This pins the corrected behaviour.
func TestDrainUncountedResultsKeepsFailureClasses(t *testing.T) {
	collector := drainTestCollector([]string{"panel-a", "panel-b", "panel-c"}, 3)
	results := make(chan fusionPanelResult, 3)
	results <- fusionPanelResult{index: 0, model: "panel-a", resp: &ModelResponse{
		Content: "a", Usage: TokenUsage{TotalTokens: 7},
	}}
	results <- fusionPanelResult{index: 1, model: "panel-b", resp: &ModelResponse{}}
	results <- fusionPanelResult{index: 2, model: "panel-c", err: assert.AnError}

	collector.drainUncountedResults(results)
	outcome := collector.outcome()

	require.Len(t, outcome.attempts, 3)
	assert.Equal(t, FusionPanelAttemptUsable, outcome.attempts[0].State)
	assert.Equal(t, FusionPanelAttemptUnusable, outcome.attempts[1].State)
	assert.Equal(t, FusionPanelAttemptFailed, outcome.attempts[2].State)
	assert.EqualValues(t, 7, outcome.usage.TotalTokens)
}
