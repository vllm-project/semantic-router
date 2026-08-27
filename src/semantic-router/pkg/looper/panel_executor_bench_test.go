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
	"fmt"
	"testing"
)

// BenchmarkPanel_RunPanel measures RunPanel's own orchestration overhead
// (goroutines, semaphore, channel, quorum bookkeeping) in isolation, with an
// in-process dispatch function that does no network I/O - the Tier-1
// counterpart to BenchmarkFusion_Execute/BenchmarkReMoM_Execute (which
// exercise RunPanel indirectly, over real localhost HTTP, and remain the
// end-to-end regression baseline for the Fusion/ReMoM migration in #2856).
func BenchmarkPanel_RunPanel(b *testing.B) {
	for _, n := range []int{1, 3, 5, 10} {
		b.Run(fmt.Sprintf("calls_%d", n), func(b *testing.B) {
			calls := panelCalls(n)
			dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
				return &ModelResponse{Model: call.Model, Content: "ok"}, nil
			}
			policy := PanelPolicy{MaxConcurrent: n, MinSuccessful: n}
			b.ReportAllocs()
			for b.Loop() {
				RunPanel(context.Background(), calls, policy, dispatch)
			}
		})
	}
}

// BenchmarkPanel_RunPanel_Quorum measures the early-return path specifically
// (quorum satisfied well before every call completes), since that's the
// path with the most bookkeeping (cancellation, early exit).
func BenchmarkPanel_RunPanel_Quorum(b *testing.B) {
	calls := panelCalls(10)
	dispatch := func(ctx context.Context, call PanelCall) (*ModelResponse, error) {
		return &ModelResponse{Model: call.Model, Content: "ok"}, nil
	}
	policy := PanelPolicy{MaxConcurrent: 10, MinSuccessful: 3, CancelOnReturn: true}
	b.ReportAllocs()
	for b.Loop() {
		RunPanel(context.Background(), calls, policy, dispatch)
	}
}
