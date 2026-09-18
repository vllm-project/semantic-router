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
	"sync"
	"time"
)

// fusionPanelBoundaries holds the two independent instants after which a panel
// result may no longer satisfy quorum.
//
// They are captured once, before any worker starts, because a context that has
// already fired cannot say when it fired. Reconstructing a boundary afterwards
// either invents one (time.Now(), which is strictly later than the event) or
// reads the wrong one (a round deadline still in the future when the caller has
// already cancelled).
type fusionPanelBoundaries struct {
	// parentDeadline is the caller's own deadline, zero when the caller set none.
	parentDeadline time.Time

	// roundDeadline is the configured round timeout, zero when unconfigured. It
	// is the exact value handed to context.WithDeadline, so the timer that stops
	// the panel and the boundary that arbitrates its results cannot disagree.
	roundDeadline time.Time
}

func newFusionPanelBoundaries(
	ctx context.Context,
	cfg fusionExecutionConfig,
	now time.Time,
) fusionPanelBoundaries {
	var bounds fusionPanelBoundaries
	if deadline, ok := ctx.Deadline(); ok {
		bounds.parentDeadline = deadline
	}
	if cfg.RoundTimeoutSeconds > 0 {
		bounds.roundDeadline = now.Add(time.Duration(cfg.RoundTimeoutSeconds) * time.Second)
	}
	return bounds
}

// crossedBy reports whether work finishing at completedAt landed past either
// configured budget. Which one is deliberately not reported: both make a result
// uncountable and both resolve to DeadlineExceeded. The rule that does
// distinguish them -- a still-connected caller stays eligible for the fallback
// after an internal round overrun -- is enforced against the caller's context in
// fusionCallerBudgetErr.
func (b fusionPanelBoundaries) crossedBy(completedAt time.Time) bool {
	// Not-before rather than after: at the deadline instant the budget is already
	// spent, so a result completing exactly then is as late as one completing
	// after. This matches fusionCallerBudgetErr, which treats the deadline
	// instant itself as exhausted.
	return !b.parentDeadline.IsZero() && !completedAt.Before(b.parentDeadline) ||
		!b.roundDeadline.IsZero() && !completedAt.Before(b.roundDeadline)
}

// newFusionPanelContext derives the panel's own context and the cancel that both
// stops in-flight attempts and releases the context on the deferred cleanup.
//
// The panel always gets a real cancel: early quorum and on_error=fail both abort
// the remaining attempts, and the join that follows would otherwise block on
// work whose result is already discarded. A configured round timeout layers the
// precomputed boundary on top so the timer and the arbiter share one instant.
func newFusionPanelContext(
	ctx context.Context,
	bounds fusionPanelBoundaries,
) (context.Context, context.CancelFunc) {
	if bounds.roundDeadline.IsZero() {
		return context.WithCancel(ctx)
	}
	return context.WithDeadline(ctx, bounds.roundDeadline)
}

// fusionPanelArbiter is the single gate every panel result passes through.
// Both select arms in the executor route through admit, because select chooses
// uniformly at random among ready cases: a rule applied on only one arm would
// leave the boundary nondeterministic.
type fusionPanelArbiter struct {
	bounds    fusionPanelBoundaries
	collector *fusionPanelCollector
	terminal  bool
	termErr   error

	// crossed records that a boundary made some result uncountable. Completion
	// timestamps prove a deadline passed while the context timer callback is
	// still pending, so the reported cause does not depend on timer delivery.
	crossed bool
}

func newFusionPanelArbiter(
	bounds fusionPanelBoundaries,
	collector *fusionPanelCollector,
) *fusionPanelArbiter {
	return &fusionPanelArbiter{bounds: bounds, collector: collector}
}

// admit applies the boundary rules in priority order and reports whether the
// panel has reached a terminal decision. A result that may not count is still
// recorded, so evidence and token usage stay complete.
func (a *fusionPanelArbiter) admit(ctx context.Context, result fusionPanelResult) bool {
	if a.terminal || a.uncountable(ctx, result) {
		a.collector.recordAttempt(result)
		return a.terminal
	}
	if done, err := a.collector.handleResult(result); done {
		a.terminal = true
		a.termErr = err
	}
	return a.terminal
}

// uncountable reports whether a result may only be accounted, never counted, and
// records which boundary rejected it.
func (a *fusionPanelArbiter) uncountable(ctx context.Context, result fusionPanelResult) bool {
	// The caller giving up outranks both deadlines. Once the parent context is
	// done nothing this panel produces can be served, so no result may move the
	// quorum -- including one that finished before the cancellation, since the
	// request it would have answered no longer exists.
	if ctx.Err() != nil {
		return true
	}
	if !a.bounds.crossedBy(result.completedAt) {
		return false
	}
	a.crossed = true
	return true
}

// drainQueued arbitrates every result already queued and reports whether the
// panel reached a terminal decision. It keeps draining past a terminal result so
// no completed attempt is dropped from evidence or usage.
func (a *fusionPanelArbiter) drainQueued(ctx context.Context, results <-chan fusionPanelResult) bool {
	for {
		select {
		case result := <-results:
			a.admit(ctx, result)
		default:
			return a.terminal
		}
	}
}

// stopAndJoin aborts the remaining attempts and waits for their goroutines.
//
// Panel goroutines touch per-execution client state that Execute resets in a
// defer once the panel returns, so every exit path must join them first. cancel
// has already aborted their in-flight calls, which keeps the join short.
//
// It deliberately does not drain: what may be done with a result that arrives
// during the join depends on why the panel stopped, and only the caller knows
// that.
func (a *fusionPanelArbiter) stopAndJoin(
	cancel context.CancelFunc,
	panelWork *sync.WaitGroup,
) {
	cancel()
	panelWork.Wait()
}

// settle stops the remaining attempts after the panel has already reached its
// own terminal decision, and accounts for whatever they had paid for.
//
// Results arriving during this join cannot be counted, because the decision they
// would have to change was already made. This is the only situation in which
// bypassing arbitration is correct.
func (a *fusionPanelArbiter) settle(
	results <-chan fusionPanelResult,
	cancel context.CancelFunc,
	panelWork *sync.WaitGroup,
) {
	a.stopAndJoin(cancel, panelWork)
	a.collector.drainUncountedResults(results)
}

// decided returns the outcome of a panel that reached its own terminal decision,
// either early quorum or an on_error=fail abort.
func (a *fusionPanelArbiter) decided() (fusionPanelOutcome, error) {
	return a.collector.outcome(), a.termErr
}

// halted returns the outcome of a panel stopped by a boundary rather than by its
// own decision.
//
// Every worker has been joined and the queue drained twice by this point, so
// each attempt already carries its own recorded state and there is nothing to
// synthesize.
func (a *fusionPanelArbiter) halted(
	cfg fusionExecutionConfig,
	cause error,
) (fusionPanelOutcome, error) {
	outcome := a.collector.outcome()
	return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, cause)
}

// exhausted returns the outcome of a panel where every attempt reported back
// without any of them tripping a terminal decision.
//
// Workers aborted by a cancellation or deadline can report back fast enough for
// this loop to consume all of them before the boundary arm is ever selected, so
// the cause is attributed here as well. Without that, the error a caller sees
// would depend on which select arm won and on when the context timer ran.
func (a *fusionPanelArbiter) exhausted(
	ctx context.Context,
	cfg fusionExecutionConfig,
) (fusionPanelOutcome, error) {
	outcome := a.collector.outcome()
	if cause := a.exhaustionCause(ctx); cause != nil {
		return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, cause)
	}
	if len(outcome.responses) < cfg.MinSuccessfulResponses {
		return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, nil)
	}
	return outcome, nil
}

// exhaustionCause prefers the caller's live error and otherwise converts a
// recorded boundary crossing, which the completion timestamps already proved,
// into the deadline error the context has not yet published.
func (a *fusionPanelArbiter) exhaustionCause(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if a.crossed {
		return context.DeadlineExceeded
	}
	return nil
}
