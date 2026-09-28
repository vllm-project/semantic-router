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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
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
	if req != nil {
		ctx = contextWithRoutingRecipe(ctx, req.RecipeName)
	}
	algorithm := looperAlgorithm(req)
	tracker, ctx, span := newAttemptTracker(ctx, algorithm)
	tracing.SetSpanAttributes(span,
		attribute.String("looper.algorithm", algorithm),
		attribute.Int("looper.candidate_count", looperCandidateCount(req)),
		attribute.Bool("looper.streaming", req != nil && req.IsStreaming),
	)

	start := time.Now()
	resp, err := l.Execute(ctx, req)
	duration := time.Since(start)
	status := "succeeded"
	if err != nil {
		status = string(classifyExecutionStatus(err))
		tracing.SetSpanAttributes(span, attribute.String("error.type", status))
		span.SetStatus(codes.Error, status)
	}
	tracing.SetSpanAttributes(span,
		attribute.String("looper.status", status),
		attribute.Int("looper.attempt_count", tracker.attemptCount()),
	)
	if resp != nil {
		resp.LatencyMs = duration.Milliseconds()
		resp.ExecutionTrace = tracker.snapshot()
		tracing.SetSpanAttributes(span,
			attribute.Int("looper.final_attempt.ordinal", resp.ExecutionTrace.FinalAttemptOrdinal),
			attribute.String("looper.final_model", resp.Model),
			attribute.Int64("looper.prompt_tokens", resp.Usage.PromptTokens),
			attribute.Int64("looper.completion_tokens", resp.Usage.CompletionTokens),
			attribute.Int64("looper.total_tokens", resp.Usage.TotalTokens),
		)
	}
	metrics.RecordLooperExecution(algorithm, status, duration.Seconds())
	span.End()
	return resp, err
}

func looperAlgorithm(req *Request) string {
	if req == nil || req.Algorithm == nil {
		return "unknown"
	}
	return req.Algorithm.Type
}

func looperCandidateCount(req *Request) int {
	if req == nil {
		return 0
	}
	return len(req.ModelRefs)
}

func classifyExecutionStatus(err error) AttemptStatus {
	status, _ := classifyAttemptError(err)
	return status
}
