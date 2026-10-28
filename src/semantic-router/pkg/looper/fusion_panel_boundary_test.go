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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func boundaryTestConfig(models []string, quorum int) fusionExecutionConfig {
	return fusionExecutionConfig{
		AnalysisModels:         models,
		MinSuccessfulResponses: quorum,
		MaxConcurrent:          len(models),
		OnError:                config.FusionOnErrorSkip,
	}
}

func usableBoundaryResult(index int, model string, completedAt time.Time) fusionPanelResult {
	return fusionPanelResult{
		index:       index,
		model:       model,
		completedAt: completedAt,
		resp:        &ModelResponse{Content: "answer", Usage: TokenUsage{TotalTokens: 10}},
	}
}

// awaitFusionPanel selects between a ready result and a fired boundary, and Go
// picks uniformly at random among ready cases. A rule enforced on only one arm
// therefore leaves the outcome to chance. Repeating the same already-expired
// panel makes it overwhelmingly unlikely that only one arm is ever taken, and
// the verdict must not vary with the arm chosen.
func TestAwaitFusionPanelRejectsLateResultOnEitherSelectArm(t *testing.T) {
	const runs = 200
	// One configured model and one enqueued result: the panel declares exactly
	// the work this harness produces, so no slot is left without a result.
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)

	for run := 0; run < runs; run++ {
		roundDeadline := time.Now().Add(-time.Second)
		bounds := fusionPanelBoundaries{roundDeadline: roundDeadline}

		panelCtx, cancel := context.WithDeadline(context.Background(), roundDeadline)
		results := make(chan fusionPanelResult, 1)
		// Completed after the round deadline, and queued before the executor
		// looks, so both select arms are ready when it does.
		results <- usableBoundaryResult(0, "panel-a", roundDeadline.Add(time.Millisecond))

		arbiter := newFusionPanelArbiter(bounds, newFusionPanelCollector(cfg, cancel))
		var panelWork sync.WaitGroup
		outcome, err := awaitFusionPanel(
			context.Background(), panelCtx, cfg, arbiter, results, cancel, &panelWork)
		cancel()

		require.Errorf(t, err, "run %d: a post-deadline response must not satisfy quorum", run)
		assert.Emptyf(t, outcome.responses, "run %d: late work must not reach the judge", run)
		// The work was still paid for, so it must survive in the accounting.
		assert.EqualValuesf(t, 10, outcome.usage.TotalTokens, "run %d: late usage must be accounted", run)
	}
}

// A result that genuinely landed before the round deadline keeps full semantics,
// so the boundary rejects late work without discarding work that beat it.
func TestAwaitFusionPanelAcceptsResultThatBeatTheRoundDeadline(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a", "panel-b"}, 1)
	roundDeadline := time.Now().Add(-time.Second)
	bounds := fusionPanelBoundaries{roundDeadline: roundDeadline}

	panelCtx, cancel := context.WithDeadline(context.Background(), roundDeadline)
	defer cancel()
	results := make(chan fusionPanelResult, 2)
	results <- usableBoundaryResult(0, "panel-a", roundDeadline.Add(-time.Millisecond))

	arbiter := newFusionPanelArbiter(bounds, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	outcome, err := awaitFusionPanel(
		context.Background(), panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.NoError(t, err)
	require.Len(t, outcome.responses, 1)
	assert.Equal(t, "answer", outcome.responses[0].Content)
}

// Cancelling the caller must end the panel even when the configured round
// deadline is still in the future. Reading the cutoff back off the panel context
// would return that future deadline and admit work the caller can no longer use.
func TestFusionPanelParentCancellationOutranksFutureRoundDeadline(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)
	bounds := fusionPanelBoundaries{
		parentDeadline: time.Now().Add(time.Hour),
		roundDeadline:  time.Now().Add(time.Hour),
	}

	ctx, cancelParent := context.WithCancel(context.Background())
	cancelParent()
	panelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make(chan fusionPanelResult, 1)
	// Well inside both deadlines, but after the caller gave up.
	results <- usableBoundaryResult(0, "panel-a", time.Now())

	arbiter := newFusionPanelArbiter(bounds, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	outcome, err := awaitFusionPanel(ctx, panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Empty(t, outcome.responses, "work completed after caller cancellation must not satisfy quorum")
	assert.EqualValues(t, 10, outcome.usage.TotalTokens)
}

// The panel must split a mixed round honestly: the response that beat the
// deadline counts, the one that did not is accounted only, and the shortfall is
// reported as a quorum failure carrying both attempts.
func TestFusionPanelMixedRoundDeadlineKeepsBothAttemptsInEvidence(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a", "panel-b"}, 2)
	roundDeadline := time.Now().Add(-time.Second)
	bounds := fusionPanelBoundaries{roundDeadline: roundDeadline}

	panelCtx, cancel := context.WithDeadline(context.Background(), roundDeadline)
	defer cancel()
	results := make(chan fusionPanelResult, 2)
	results <- usableBoundaryResult(0, "panel-a", roundDeadline.Add(-time.Millisecond))
	results <- usableBoundaryResult(1, "panel-b", roundDeadline.Add(time.Millisecond))

	arbiter := newFusionPanelArbiter(bounds, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	outcome, err := awaitFusionPanel(
		context.Background(), panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.Error(t, err)
	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, 2, evidence.RequiredCount)
	assert.Equal(t, 1, evidence.UsableCount, "only the response that beat the deadline may count")
	// Both attempts were real calls, so both must appear with their usage.
	require.Len(t, evidence.Attempts, 2)
	assert.Equal(t, FusionPanelAttemptUsable, evidence.Attempts[0].State)
	assert.Equal(t, FusionPanelAttemptUsable, evidence.Attempts[1].State)
	assert.EqualValues(t, 20, evidence.Usage.TotalTokens)
	assert.Len(t, outcome.responses, 1)
}

// End to end through Execute: a caller who cancels mid-panel gets a cancellation
// error, and neither the judge nor the quorum fallback is allowed to spend more
// budget on a request nobody is waiting for.
func TestFusionLooperCallerCancellationSkipsJudgeAndFallback(t *testing.T) {
	var judgeCalls, fallbackCalls atomic.Int64
	release := make(chan struct{})
	panelStarted := make(chan struct{}, 3)

	server := newFusionStubServer(t, func(model, _ string) (string, int) {
		switch model {
		case "judge":
			judgeCalls.Add(1)
			return "unexpected synthesis", http.StatusOK
		case "fallback-target":
			fallbackCalls.Add(1)
			return "unexpected fallback", http.StatusOK
		default:
			panelStarted <- struct{}{}
			<-release
			return "panel answer", http.StatusOK
		}
	})
	defer server.Close()
	defer close(release)

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b"},
			MaxConcurrent:          2,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
			QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:   "fallback-target",
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel only once the panel is demonstrably in flight, so the test covers
	// cancellation during panel work rather than before routing began.
	go func() {
		<-panelStarted
		cancel()
	}()

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(ctx, req)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Zero(t, judgeCalls.Load(), "judge must not run for a cancelled request")
	assert.Zero(t, fallbackCalls.Load(), "fallback must not spend budget after the caller gave up")
}

// When every worker reports back before the boundary arm is selected, the loop
// exits through its own exhaustion path rather than through panelCtx.Done. A
// cancelled caller must still produce a cancellation error there, otherwise the
// error the caller sees depends on which select arm happened to win.
func TestFusionPanelExhaustedLoopStillReportsCallerCancellation(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)

	ctx, cancelParent := context.WithCancel(context.Background())
	cancelParent()
	panelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	results := make(chan fusionPanelResult, 1)
	results <- usableBoundaryResult(0, "panel-a", time.Now())

	arbiter := newFusionPanelArbiter(fusionPanelBoundaries{}, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	_, err := awaitFusionPanel(ctx, panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// elapsedDeadlineContext models the window every deadline-bearing context passes
// through: the wall-clock deadline has passed, but the runtime has not yet run
// the timer callback, so Done is still open and Err still reports nil.
//
// Reading only Err in that window makes the decision depend on timer delivery,
// which is exactly the nondeterminism the timestamp arbiter exists to remove.
type elapsedDeadlineContext struct {
	context.Context
	deadline time.Time
}

func (c elapsedDeadlineContext) Deadline() (time.Time, bool) { return c.deadline, true }
func (c elapsedDeadlineContext) Done() <-chan struct{}       { return nil }
func (c elapsedDeadlineContext) Err() error                  { return nil }

func newElapsedDeadlineContext() context.Context {
	return elapsedDeadlineContext{Context: context.Background(), deadline: time.Now().Add(-time.Second)}
}

// When every worker reports a late result through the result arm, the loop exits
// through exhaustion. The deadline that rejected those results must still reach
// the caller, rather than being lost because the context timer had not fired.
func TestFusionPanelExhaustedLoopPreservesRoundDeadlineCause(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)
	roundDeadline := time.Now().Add(-time.Second)

	// panelCtx deliberately never signals, forcing the result arm to win.
	panelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	results := make(chan fusionPanelResult, 1)
	results <- usableBoundaryResult(0, "panel-a", roundDeadline.Add(time.Millisecond))

	arbiter := newFusionPanelArbiter(
		fusionPanelBoundaries{roundDeadline: roundDeadline}, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	_, err := awaitFusionPanel(
		context.Background(), panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// The same for a crossed parent deadline, which must not be reported as a plain
// below-quorum failure just because the timer callback is lagging.
func TestFusionPanelExhaustedLoopPreservesParentDeadlineCause(t *testing.T) {
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)
	ctx := newElapsedDeadlineContext()
	parentDeadline, _ := ctx.Deadline()

	panelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	results := make(chan fusionPanelResult, 1)
	results <- usableBoundaryResult(0, "panel-a", parentDeadline.Add(time.Millisecond))

	arbiter := newFusionPanelArbiter(
		fusionPanelBoundaries{parentDeadline: parentDeadline}, newFusionPanelCollector(cfg, cancel))
	var panelWork sync.WaitGroup
	_, err := awaitFusionPanel(ctx, panelCtx, cfg, arbiter, results, cancel, &panelWork)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

// Either configured budget makes a late result uncountable, and an unconfigured
// budget never does. A zero deadline is the "unset" marker, so treating it as an
// instant in 1970 would make the panel discard every result it ever produced.
func TestFusionPanelBoundaryCrossedByHonoursOnlyConfiguredBudgets(t *testing.T) {
	now := time.Now()
	before := now.Add(-time.Millisecond)
	after := now.Add(time.Millisecond)

	none := fusionPanelBoundaries{}
	parentOnly := fusionPanelBoundaries{parentDeadline: now}
	roundOnly := fusionPanelBoundaries{roundDeadline: now}
	both := fusionPanelBoundaries{parentDeadline: now, roundDeadline: now.Add(time.Minute)}

	assert.False(t, none.crossedBy(after), "an unbounded panel has nothing to overrun")
	assert.False(t, parentOnly.crossedBy(before))
	assert.True(t, parentOnly.crossedBy(after))
	assert.False(t, roundOnly.crossedBy(before))
	assert.True(t, roundOnly.crossedBy(after))

	// The deadline instant itself is already spent, so a result completing
	// exactly then is late. Treating it as in time would let a panel satisfy a
	// quorum out of a budget it had fully consumed.
	assert.True(t, parentOnly.crossedBy(now), "the parent deadline instant is exhausted")
	assert.True(t, roundOnly.crossedBy(now), "the round deadline instant is exhausted")

	// The earlier of the two budgets is what actually stops the panel.
	assert.True(t, both.crossedBy(after), "overrunning either budget is enough")
	assert.False(t, both.crossedBy(before))
}

// settlementRaceHarness runs awaitFusionPanel with one worker that holds a
// result until settlement has begun, then enqueues it.
//
// The send therefore always lands after the boundary arm has drained the queue
// and entered the join, which is the window where countability used to depend on
// scheduling rather than on the result's own timestamp.
func settlementRaceHarness(
	t *testing.T,
	roundDeadline time.Time,
	stampedAt time.Time,
) (fusionPanelOutcome, error) {
	t.Helper()
	cfg := boundaryTestConfig([]string{"panel-a"}, 1)

	panelCtx, cancelPanel := context.WithDeadline(context.Background(), roundDeadline)
	defer cancelPanel()

	results := make(chan fusionPanelResult, 1)

	var panelWork sync.WaitGroup
	panelWork.Add(1)
	settlementBegun := make(chan struct{})
	go func() {
		defer panelWork.Done()
		<-settlementBegun
		results <- usableBoundaryResult(0, "panel-a", stampedAt)
	}()

	// stopAndJoin cancels before it joins, so releasing the worker here guarantees
	// the send happens during the join rather than before it.
	releaseOnCancel := func() {
		select {
		case <-settlementBegun:
		default:
			close(settlementBegun)
		}
		cancelPanel()
	}

	arbiter := newFusionPanelArbiter(
		fusionPanelBoundaries{roundDeadline: roundDeadline},
		newFusionPanelCollector(cfg, releaseOnCancel))

	return awaitFusionPanel(
		context.Background(), panelCtx, cfg, arbiter, results, releaseOnCancel, &panelWork)
}

// A result stamped before the cutoff still satisfies quorum when its channel
// send lands during settlement.
//
// The worker finished its call in time; only the handoff was late. Discarding it
// would fail a request whose quorum was genuinely met, and would make the
// outcome depend on goroutine scheduling.
func TestFusionPanelCountsPreCutoffResultEnqueuedDuringSettlement(t *testing.T) {
	roundDeadline := time.Now().Add(-time.Second)

	outcome, err := settlementRaceHarness(t, roundDeadline, roundDeadline.Add(-time.Millisecond))

	require.NoError(t, err, "a pre-cutoff result must satisfy quorum even when it arrives during the join")
	require.Len(t, outcome.responses, 1)
	assert.Equal(t, "answer", outcome.responses[0].Content,
		"the in-time response must be the one that satisfied quorum")
}

// The converse, so the fix above cannot degrade into accepting anything that
// arrives during settlement: a post-cutoff result is still rejected.
func TestFusionPanelRejectsPostCutoffResultEnqueuedDuringSettlement(t *testing.T) {
	roundDeadline := time.Now().Add(-time.Second)

	_, err := settlementRaceHarness(t, roundDeadline, roundDeadline.Add(time.Millisecond))

	require.Error(t, err, "a post-cutoff result must not satisfy the quorum the boundary resolved")
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}
