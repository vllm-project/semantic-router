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
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

func panelCalls(n int) []PanelCall {
	calls := make([]PanelCall, n)
	for i := range calls {
		calls[i] = PanelCall{Index: i, Model: fmt.Sprintf("model-%d", i)}
	}
	return calls
}

func TestRunPanelEmptyCallsReturnsImmediately(t *testing.T) {
	result := RunPanel(context.Background(), nil, PanelPolicy{}, func(context.Context, PanelCall) (*ModelResponse, error) {
		t.Fatal("dispatch must not be called for an empty panel")
		return nil, nil
	})
	if !result.QuorumMet {
		t.Fatalf("QuorumMet = false, want true for a trivially-satisfied empty quorum")
	}
	if len(result.Attempts) != 0 {
		t.Fatalf("Attempts = %v, want empty", result.Attempts)
	}
}

func TestRunPanelFullDrainPreservesIndexOrder(t *testing.T) {
	calls := panelCalls(5)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		// Reverse-order completion: highest index finishes first.
		time.Sleep(time.Duration(5-call.Index) * 5 * time.Millisecond)
		return &ModelResponse{Model: call.Model, Content: call.Model}, nil
	}
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 5, MinSuccessful: 5}, dispatch)
	if !result.QuorumMet {
		t.Fatalf("QuorumMet = false, want true")
	}
	successful := result.Successful()
	if len(successful) != 5 {
		t.Fatalf("Successful() length = %d, want 5", len(successful))
	}
	for i, resp := range successful {
		want := fmt.Sprintf("model-%d", i)
		if resp.Model != want {
			t.Fatalf("Successful()[%d].Model = %q, want %q (order must be by call index, not completion order)", i, resp.Model, want)
		}
	}
}

// A straggler that hasn't even been scheduled to start by the time quorum
// is met is legitimately never dispatched at all (its outer semaphore/done
// select can validly pick the done branch before calling dispatch) - that's
// correct, efficient behavior, not something to assert against. So this
// test uses a start barrier: every straggler signals the instant it enters
// dispatch, and the fast calls don't return (satisfying quorum) until all
// stragglers have checked in - guaranteeing dispatch really was called for
// each one, so cancellation-of-in-flight-work is what's actually verified.
func TestRunPanelQuorumMetReturnsEarlyAndCancelsStragglers(t *testing.T) {
	calls := panelCalls(6)
	const stragglerCount = 4
	started := make(chan int, stragglerCount)
	cancelled := make(chan int, stragglerCount)
	allStragglersStarted := make(chan struct{})
	go func() {
		seen := map[int]bool{}
		for len(seen) < stragglerCount {
			seen[<-started] = true
		}
		close(allStragglersStarted)
	}()

	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		if call.Index < 2 {
			<-allStragglersStarted // fast calls wait so quorum can't fire early
			return &ModelResponse{Model: call.Model}, nil
		}
		started <- call.Index
		select {
		case <-time.After(2 * time.Second):
			return &ModelResponse{Model: call.Model}, nil
		case <-ctx.Done():
			cancelled <- call.Index
			return nil, ctx.Err()
		}
	}
	start := time.Now()
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 6, MinSuccessful: 2, CancelOnReturn: true}, dispatch)
	elapsed := time.Since(start)

	if !result.QuorumMet {
		t.Fatalf("QuorumMet = false, want true")
	}
	if elapsed > time.Second {
		t.Fatalf("RunPanel took %v to return after quorum was met, want well under 1s (stragglers should be cancelled)", elapsed)
	}
	seen := map[int]bool{}
	timeout := time.After(4 * time.Second)
	for len(seen) < stragglerCount {
		select {
		case idx := <-cancelled:
			seen[idx] = true
		case <-timeout:
			t.Fatalf("only %d/%d in-flight stragglers reported cancellation within 4s: %v", len(seen), stragglerCount, seen)
		}
	}
}

func TestRunPanelQuorumMetWithoutCancelLetsStragglersFinishInBackground(t *testing.T) {
	calls := panelCalls(4)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		if call.Index < 2 {
			return &ModelResponse{Model: call.Model}, nil
		}
		time.Sleep(150 * time.Millisecond) // deliberately not ctx-aware: must not leak or panic
		return &ModelResponse{Model: call.Model}, nil
	}
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 4, MinSuccessful: 2, CancelOnReturn: false}, dispatch)
	if !result.QuorumMet {
		t.Fatalf("QuorumMet = false, want true")
	}
	// Give the non-cancellable stragglers time to actually finish so they
	// don't outlive the test (goroutine-leak coverage lives in
	// panel_executor_leak_test.go; this just avoids flaking that check).
	time.Sleep(300 * time.Millisecond)
}

func TestRunPanelFailFastAbortsOnFirstFailure(t *testing.T) {
	calls := panelCalls(5)
	boom := errors.New("boom")
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		if call.Index == 0 {
			return nil, boom
		}
		select {
		case <-time.After(2 * time.Second):
			return &ModelResponse{Model: call.Model}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	start := time.Now()
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 5, MinSuccessful: 5, FailFast: true, CancelOnReturn: true}, dispatch)
	elapsed := time.Since(start)

	if !errors.Is(result.Err, boom) {
		t.Fatalf("Err = %v, want %v", result.Err, boom)
	}
	if result.QuorumMet {
		t.Fatalf("QuorumMet = true, want false (fail-fast aborted before quorum)")
	}
	if elapsed > time.Second {
		t.Fatalf("RunPanel took %v to return after fail-fast, want well under 1s", elapsed)
	}
}

// Quorum of 3 is unreachable (index 1 always fails, only 2 calls can ever
// succeed), so this deterministically forces a full natural drain rather
// than racing an early return against which results happen to be observed
// first - every attempt is guaranteed to be seen before RunPanel returns.
func TestRunPanelSkipModeContinuesPastFailures(t *testing.T) {
	calls := panelCalls(3)
	boom := errors.New("boom")
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		if call.Index == 1 {
			return nil, boom
		}
		return &ModelResponse{Model: call.Model}, nil
	}
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 3, MinSuccessful: 3, FailFast: false}, dispatch)
	if result.QuorumMet {
		t.Fatalf("QuorumMet = true, want false (only 2 of 3 required successes are possible)")
	}
	if result.Err != nil {
		t.Fatalf("Err = %v, want nil in skip mode", result.Err)
	}
	if got := len(result.Successful()); got != 2 {
		t.Fatalf("Successful() length = %d, want 2 (skip mode must continue past the index-1 failure)", got)
	}
	failed := result.Failed()
	if len(failed) != 1 || failed[0].Index != 1 || !errors.Is(failed[0].Err, boom) {
		t.Fatalf("Failed() = %+v, want exactly the index-1 failure", failed)
	}
}

func TestRunPanelTimeoutReturnsPromptlyNotAfterSlowestStraggler(t *testing.T) {
	calls := panelCalls(4)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		select {
		case <-time.After(5 * time.Second):
			return &ModelResponse{Model: call.Model}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	start := time.Now()
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 4, MinSuccessful: 4, Timeout: 80 * time.Millisecond, CancelOnReturn: true}, dispatch)
	elapsed := time.Since(start)

	if !result.TimedOut {
		t.Fatalf("TimedOut = false, want true")
	}
	if !errors.Is(result.Err, context.DeadlineExceeded) {
		t.Fatalf("Err = %v, want context.DeadlineExceeded", result.Err)
	}
	if elapsed > time.Second {
		t.Fatalf("RunPanel took %v to return after its 80ms timeout, want well under 1s (must not wait for the 5s stragglers)", elapsed)
	}
}

func TestRunPanelMinSuccessfulGreaterThanCallCountDrainsWithoutError(t *testing.T) {
	calls := panelCalls(3)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		return &ModelResponse{Model: call.Model}, nil
	}
	result := RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 3, MinSuccessful: 10}, dispatch)
	if result.QuorumMet {
		t.Fatalf("QuorumMet = true, want false (quorum of 10 is unreachable with only 3 calls)")
	}
	if result.Err != nil {
		t.Fatalf("Err = %v, want nil (unreachable quorum is not itself an error)", result.Err)
	}
	if len(result.Successful()) != 3 {
		t.Fatalf("Successful() length = %d, want 3", len(result.Successful()))
	}
}

func TestRunPanelMaxConcurrentLimitsInFlightDispatches(t *testing.T) {
	calls := panelCalls(8)
	var inFlight int32
	var maxObserved int32
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		current := atomic.AddInt32(&inFlight, 1)
		for {
			observed := atomic.LoadInt32(&maxObserved)
			if current <= observed || atomic.CompareAndSwapInt32(&maxObserved, observed, current) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		return &ModelResponse{Model: call.Model}, nil
	}
	RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 2, MinSuccessful: 8}, dispatch)
	if got := atomic.LoadInt32(&maxObserved); got > 2 {
		t.Fatalf("max concurrent dispatches observed = %d, want <= 2", got)
	}
}

// Only index 2 counts toward quorum, so a quorum of 2 is unreachable and
// RunPanel is guaranteed to drain all 3 calls before returning -
// deterministic, unlike a reachable quorum racing which results land first.
func TestRunPanelCustomCountsTowardQuorumDoesNotFilterSuccessful(t *testing.T) {
	calls := panelCalls(3)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		content := ""
		if call.Index == 2 {
			content = "usable"
		}
		return &ModelResponse{Model: call.Model, Content: content}, nil
	}
	usableOnly := func(a PanelAttempt) bool {
		return a.Outcome == PanelOutcomeSuccess && a.Response.Content != ""
	}
	result := RunPanel(context.Background(), calls, PanelPolicy{
		MaxConcurrent: 3, MinSuccessful: 2, CountsTowardQuorum: usableOnly,
	}, dispatch)
	if result.QuorumMet {
		t.Fatalf("QuorumMet = true, want false (only 1 attempt ever counts toward quorum=2)")
	}
	// All 3 dispatch successfully even though only 1 counts toward quorum -
	// Successful() must still surface every successful (non-error) response,
	// since callers like ReMoM need the full set for token accounting even
	// when only some are "usable" for synthesis.
	successful := result.Successful()
	if len(successful) != 3 {
		t.Fatalf("Successful() length = %d, want 3 (quorum predicate must not filter Successful())", len(successful))
	}
	if successful[2].Content != "usable" {
		t.Fatalf("Successful()[2].Content = %q, want %q", successful[2].Content, "usable")
	}
}

func TestPanelResultNilSafety(t *testing.T) {
	var result *PanelResult
	if got := result.Successful(); got != nil {
		t.Fatalf("nil PanelResult.Successful() = %v, want nil", got)
	}
	if got := result.Failed(); got != nil {
		t.Fatalf("nil PanelResult.Failed() = %v, want nil", got)
	}
}
