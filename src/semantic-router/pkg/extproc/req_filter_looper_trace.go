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

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// looperReplayDiagnostics projects runtime attempt evidence into the Replay schema.
func looperReplayDiagnostics(trace looper.ExecutionTrace) *routerreplay.LooperDiagnostics {
	if trace.Version == 0 {
		return nil
	}
	diagnostics := &routerreplay.LooperDiagnostics{
		Version:             trace.Version,
		TraceID:             trace.TraceID,
		Algorithm:           trace.Algorithm,
		FinalAttemptOrdinal: trace.FinalAttemptOrdinal,
		AttemptsTruncated:   trace.AttemptsTruncated,
		DroppedAttemptCount: trace.DroppedAttemptCount,
		DroppedUsage:        replayLooperUsage(trace.DroppedUsage),
		Attempts:            make([]routerreplay.LooperAttempt, len(trace.Attempts)),
	}
	for index, attempt := range trace.Attempts {
		diagnostics.Attempts[index] = routerreplay.LooperAttempt{
			Ordinal: attempt.Ordinal, Stage: attempt.Stage,
			Role: attempt.Role, Model: attempt.Model, Status: string(attempt.Status),
			Reason: string(attempt.Reason), Usable: cloneReplayValue(attempt.Usable),
			Accepted: cloneReplayValue(attempt.Accepted), Selected: attempt.Selected,
			Synthesized: attempt.Synthesized, Discarded: attempt.Discarded,
			VerifierType: attempt.VerifierType, VerifierVersion: attempt.VerifierVersion,
			Score: cloneReplayValue(attempt.Score), Threshold: cloneReplayValue(attempt.Threshold),
			ReservedTokens:                   cloneReplayValue(attempt.ReservedTokens),
			EstimatedTokens:                  cloneReplayValue(attempt.EstimatedTokens),
			EffectiveMaxOutputTokens:         cloneReplayValue(attempt.EffectiveMaxOutputTokens),
			EffectiveMaxOutputTokensSource:   attempt.EffectiveMaxOutputTokensSource,
			EffectiveMaxOutputTokensFallback: attempt.EffectiveMaxOutputTokensFallback,
			Usage:                            replayLooperUsage(attempt.Usage),
			EstimatedCost:                    cloneReplayValue(attempt.EstimatedCost),
			ActualCost:                       cloneReplayValue(attempt.ActualCost), Currency: attempt.Currency,
			QueueLatencyMs:     cloneReplayValue(attempt.QueueLatencyMs),
			FirstByteLatencyMs: cloneReplayValue(attempt.FirstByteLatencyMs),
			TotalLatencyMs:     attempt.TotalLatencyMs,
		}
	}
	return diagnostics
}

func replayLooperUsage(usage looper.TokenUsage) routerreplay.LooperUsage {
	return routerreplay.LooperUsage{
		PromptTokens: usage.PromptTokens, CompletionTokens: usage.CompletionTokens, TotalTokens: usage.TotalTokens,
	}
}

func cloneReplayValue[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
