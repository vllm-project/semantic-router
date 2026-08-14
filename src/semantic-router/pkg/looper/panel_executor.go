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

// PanelCall is one candidate dispatched as part of a concurrent panel. Index
// is the deterministic result-slot the caller wants this candidate's outcome
// written to, independent of completion order.
type PanelCall struct {
	Index int
	Model string
}

// PanelDispatchFunc performs one candidate's call. It must respect ctx
// cancellation - every Looper algorithm's underlying Client.CallModel
// already does, since it is passed ctx directly.
type PanelDispatchFunc func(ctx context.Context, call PanelCall) (*ModelResponse, error)

// PanelOutcome is a stable, typed classification of one attempt's result.
type PanelOutcome string

const (
	// PanelOutcomeSuccess means dispatch returned a response with no error.
	PanelOutcomeSuccess PanelOutcome = "success"
	// PanelOutcomeFailed means dispatch returned an error (including a
	// dispatch-supplied "invalid response" error synthesized by the caller).
	PanelOutcomeFailed PanelOutcome = "failed"
	// PanelOutcomeCancelled means RunPanel stopped waiting before this
	// call's dispatch was ever observed to complete - either it was still
	// queued behind the concurrency limit, or it was in flight, when the
	// executor returned. The underlying goroutine is still guaranteed to
	// terminate on its own (see RunPanel's leak-safety notes); this outcome
	// only reflects that the caller chose not to wait for it.
	PanelOutcomeCancelled PanelOutcome = "cancelled"
)

// PanelAttempt is one candidate's outcome.
type PanelAttempt struct {
	Index    int
	Model    string
	Response *ModelResponse
	Err      error
	Outcome  PanelOutcome
}

// PanelPolicy configures one RunPanel invocation. All fields are optional;
// zero values fall back to "require everything, no early exit" behavior.
type PanelPolicy struct {
	// MaxConcurrent caps in-flight dispatches. <=0 means len(calls) (no cap).
	MaxConcurrent int

	// MinSuccessful is the quorum: once this many attempts count toward
	// quorum (see CountsTowardQuorum), RunPanel returns immediately without
	// waiting for the rest. <=0 means len(calls) (wait for all). A value
	// greater than len(calls) is used as given, not clamped down - quorum
	// is then genuinely unreachable and RunPanel drains all calls naturally
	// (not an error by itself). Callers that want quorum clamped to
	// len(calls) instead should clamp before constructing PanelPolicy.
	MinSuccessful int

	// Timeout bounds the whole panel's wall-clock duration. <=0 means no
	// timeout (only ctx's own deadline, if any, applies).
	Timeout time.Duration

	// FailFast, when true, makes RunPanel return immediately (with Err set)
	// on the first failed attempt, mirroring each algorithm's on_error=fail
	// config. When false ("skip" mode), failures are recorded and dispatch
	// continues toward quorum or full completion.
	FailFast bool

	// CancelOnReturn, when true, cancels every remaining in-flight or
	// queued call the instant RunPanel decides to return early (quorum met
	// or FailFast triggered). When false, those calls keep running in the
	// background - RunPanel does not wait for them, but they are not
	// preempted either.
	CancelOnReturn bool

	// CountsTowardQuorum decides whether one attempt counts toward
	// MinSuccessful. Defaults to Outcome==PanelOutcomeSuccess. Algorithms
	// with a secondary usability check beyond "the call didn't error"
	// (e.g. ReMoM's usable-content filter) override this.
	CountsTowardQuorum func(PanelAttempt) bool
}

// PanelResult is the structured outcome of one RunPanel call.
type PanelResult struct {
	// Attempts is always len(calls) long and index-ordered - Attempts[i]
	// corresponds to the call with Index==i, regardless of completion
	// order. Slots RunPanel never observed a result for before returning
	// early are PanelOutcomeCancelled.
	Attempts []PanelAttempt

	// QuorumMet reports whether MinSuccessful was reached.
	QuorumMet bool

	// TimedOut reports whether Timeout (or the caller's own ctx) elapsed
	// before quorum was reached.
	TimedOut bool

	// Err is set when FailFast aborted the panel, or when the panel timed
	// out. It is nil when quorum was met or the panel drained naturally
	// (regardless of how many individual attempts failed along the way).
	Err error
}

// Successful returns every successful response, in call order (Index
// order), compacted - i.e. exactly the shape every current algorithm's
// panel-collection code already returns downstream.
func (r *PanelResult) Successful() []*ModelResponse {
	if r == nil {
		return nil
	}
	responses := make([]*ModelResponse, 0, len(r.Attempts))
	for _, attempt := range r.Attempts {
		if attempt.Outcome == PanelOutcomeSuccess {
			responses = append(responses, attempt.Response)
		}
	}
	return responses
}

// Failed returns every attempt that was actually dispatched and failed,
// excluding cancelled (never-observed) attempts.
func (r *PanelResult) Failed() []PanelAttempt {
	if r == nil {
		return nil
	}
	var failed []PanelAttempt
	for _, attempt := range r.Attempts {
		if attempt.Outcome == PanelOutcomeFailed {
			failed = append(failed, attempt)
		}
	}
	return failed
}

type panelRawResult struct {
	index     int
	model     string
	resp      *ModelResponse
	err       error
	cancelled bool
}

// RunPanel dispatches calls concurrently under policy and returns their
// structured, deterministically-ordered outcome. It never leaks goroutines:
// the results channel is buffered to len(calls) so every dispatched
// goroutine's single send always succeeds immediately regardless of whether
// RunPanel is still reading, and every goroutine terminates on its own
// (either via the pre-dispatch ctx.Done() branch, or because dispatch
// itself must return once execCtx is cancelled) independent of a background
// sync.WaitGroup that only exists for channel-close hygiene.
func RunPanel(ctx context.Context, calls []PanelCall, policy PanelPolicy, dispatch PanelDispatchFunc) *PanelResult {
	n := len(calls)
	if n == 0 {
		return &PanelResult{QuorumMet: policy.MinSuccessful <= 0}
	}

	execCtx, cancel := context.WithCancel(ctx)
	if policy.Timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, policy.Timeout)
	}
	defer cancel()

	sem := make(chan struct{}, clampPanelInt(policy.MaxConcurrent, 1, n))
	results := make(chan panelRawResult, n)

	var wg sync.WaitGroup
	for _, call := range calls {
		wg.Add(1)
		go func(call PanelCall) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-execCtx.Done():
				results <- panelRawResult{index: call.Index, model: call.Model, err: execCtx.Err(), cancelled: true}
				return
			}
			defer func() { <-sem }()
			resp, err := dispatch(execCtx, call)
			results <- panelRawResult{index: call.Index, model: call.Model, resp: resp, err: err}
		}(call)
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	return collectPanelResults(execCtx, cancel, n, calls, policy, results)
}

func collectPanelResults(
	execCtx context.Context,
	cancel context.CancelFunc,
	n int,
	calls []PanelCall,
	policy PanelPolicy,
	results <-chan panelRawResult,
) *PanelResult {
	countsTowardQuorum := policy.CountsTowardQuorum
	if countsTowardQuorum == nil {
		countsTowardQuorum = func(a PanelAttempt) bool { return a.Outcome == PanelOutcomeSuccess }
	}
	// MinSuccessful is used as given, not clamped to n: callers that want
	// clamping (ReMoM, Workflows) apply their own existing helper before
	// constructing PanelPolicy; Fusion's current config is used raw and can
	// leave quorum genuinely unreachable, which RunPanel preserves exactly
	// (a natural full drain, not an error) rather than silently clamping it.
	minSuccessful := policy.MinSuccessful
	if minSuccessful <= 0 {
		minSuccessful = n
	}

	attempts := make([]PanelAttempt, n)
	for i, call := range calls {
		attempts[i] = PanelAttempt{Index: call.Index, Model: call.Model, Outcome: PanelOutcomeCancelled}
	}
	result := &PanelResult{Attempts: attempts}

	successCount := 0
	remaining := n
	for remaining > 0 {
		select {
		case raw, ok := <-results:
			if !ok {
				remaining = 0
				continue
			}
			remaining--
			attempt := classifyPanelAttempt(raw)
			attempts[raw.index] = attempt

			if countsTowardQuorum(attempt) {
				successCount++
			}
			if successCount >= minSuccessful {
				result.QuorumMet = true
				if policy.CancelOnReturn {
					cancel()
				}
				return result
			}
			if attempt.Outcome == PanelOutcomeFailed && policy.FailFast {
				if policy.CancelOnReturn {
					cancel()
				}
				result.Err = attempt.Err
				return result
			}
		case <-execCtx.Done():
			result.TimedOut = true
			result.Err = execCtx.Err()
			if policy.CancelOnReturn {
				cancel()
			}
			return result
		}
	}
	return result
}

func classifyPanelAttempt(raw panelRawResult) PanelAttempt {
	outcome := PanelOutcomeSuccess
	switch {
	case raw.cancelled:
		outcome = PanelOutcomeCancelled
	case raw.err != nil:
		outcome = PanelOutcomeFailed
	}
	return PanelAttempt{Index: raw.index, Model: raw.model, Response: raw.resp, Err: raw.err, Outcome: outcome}
}

func clampPanelInt(v, lo, hi int) int {
	if v <= 0 || v > hi {
		return hi
	}
	if v < lo {
		return lo
	}
	return v
}
