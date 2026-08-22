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
	"testing"
	"time"

	"go.uber.org/goleak"
)

// TestRunPanelDoesNotLeakGoroutines exercises every early-return path
// (quorum met, quorum met without cancel, fail-fast, timeout) plus full
// natural drain, and asserts via goleak that no goroutine RunPanel spawned
// is still alive once the test finishes - the direct acceptance criterion
// issue #2856 calls for. goleak.VerifyNone polls with backoff internally,
// so non-cancelled stragglers that are still winding down in the background
// (the CancelOnReturn=false case) are given time to actually finish rather
// than producing a false positive.
func TestRunPanelDoesNotLeakGoroutines(t *testing.T) {
	defer goleak.VerifyNone(t, goleak.IgnoreCurrent())

	t.Run("quorum met with cancel", func(t *testing.T) {
		calls := panelCalls(6)
		dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
			if call.Index < 2 {
				return &ModelResponse{Model: call.Model}, nil
			}
			select {
			case <-time.After(2 * time.Second):
				return &ModelResponse{Model: call.Model}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 6, MinSuccessful: 2, CancelOnReturn: true}, dispatch)
	})

	t.Run("quorum met without cancel lets non-ctx-aware stragglers finish", func(t *testing.T) {
		calls := panelCalls(4)
		dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
			if call.Index < 2 {
				return &ModelResponse{Model: call.Model}, nil
			}
			time.Sleep(120 * time.Millisecond) // deliberately not ctx-aware
			return &ModelResponse{Model: call.Model}, nil
		}
		RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 4, MinSuccessful: 2, CancelOnReturn: false}, dispatch)
	})

	t.Run("fail fast with cancel", func(t *testing.T) {
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
		RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 5, MinSuccessful: 5, FailFast: true, CancelOnReturn: true}, dispatch)
	})

	t.Run("timeout with ctx-aware stragglers", func(t *testing.T) {
		calls := panelCalls(4)
		dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
			select {
			case <-time.After(5 * time.Second):
				return &ModelResponse{Model: call.Model}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 4, MinSuccessful: 4, Timeout: 60 * time.Millisecond, CancelOnReturn: true}, dispatch)
	})

	t.Run("full drain with limited concurrency", func(t *testing.T) {
		calls := panelCalls(8)
		dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
			time.Sleep(5 * time.Millisecond)
			return &ModelResponse{Model: call.Model}, nil
		}
		RunPanel(context.Background(), calls, PanelPolicy{MaxConcurrent: 2, MinSuccessful: 8}, dispatch)
	})

	// Let anything still winding down (notably the non-cancelled stragglers
	// above) actually finish before goleak.VerifyNone runs at the end.
	time.Sleep(300 * time.Millisecond)
}
