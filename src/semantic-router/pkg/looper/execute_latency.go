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
	"time"
)

// ExecuteWithLatency runs l.Execute and records the wall-clock latency of the
// full execution, in milliseconds, on the returned Response's LatencyMs
// field. It is the single instrumentation point for every Looper
// implementation (base, ratings, fusion, confidence, remom, workflows,
// rl_driven), so wall-clock latency is captured consistently regardless of
// whether the underlying algorithm dispatches its model calls sequentially or
// concurrently (issue #2694). A nil Response is passed through unchanged: no
// current implementation returns (nil, nil), but Execute's contract does not
// forbid it, so this avoids a nil-pointer panic if one ever does.
func ExecuteWithLatency(ctx context.Context, l Looper, req *Request) (*Response, error) {
	start := time.Now()
	resp, err := l.Execute(ctx, req)
	if err != nil || resp == nil {
		return resp, err
	}
	resp.LatencyMs = time.Since(start).Milliseconds()
	return resp, nil
}
