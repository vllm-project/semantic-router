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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestLooperReplayDiagnosticsProjectsContentFreeTrace(t *testing.T) {
	accepted := true
	score := 0.9
	diagnostics := looperReplayDiagnostics(looper.ExecutionTrace{
		Version:             looper.ExecutionTraceVersion,
		TraceID:             "trace-1",
		Algorithm:           "confidence",
		FinalAttemptOrdinal: 1,
		Attempts: []looper.AttemptTrace{{
			Ordinal: 1, Stage: "candidate", Role: "generator",
			Model: "large", Status: looper.AttemptStatusSucceeded,
			Reason: looper.AttemptReasonThresholdMet, Accepted: &accepted, Selected: true,
			Score: &score, Usage: looper.TokenUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
		}},
	})
	if diagnostics == nil || diagnostics.TraceID != "trace-1" || diagnostics.FinalAttemptOrdinal != 1 || len(diagnostics.Attempts) != 1 {
		t.Fatalf("projected diagnostics = %+v", diagnostics)
	}
	attempt := diagnostics.Attempts[0]
	if !attempt.Selected || attempt.Accepted == nil || !*attempt.Accepted || attempt.Usage.TotalTokens != 5 {
		t.Fatalf("projected attempt = %+v", attempt)
	}
}
