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

package store

import "testing"

func TestCloneRecordClonesLooperDiagnostics(t *testing.T) {
	accepted := true
	record := Record{RouteDiagnostics: &RouteDiagnostics{Looper: &LooperDiagnostics{
		Version:   1,
		TraceID:   "trace-1",
		Algorithm: "confidence",
		Attempts: []LooperAttempt{{
			Ordinal: 1, Stage: "candidate",
			Status: "succeeded", Accepted: &accepted, Selected: true,
			Usage: LooperUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
		}},
		FinalAttemptOrdinal: 1,
	}}}

	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.Looper.Attempts[0].Model = "changed"
	*cloned.RouteDiagnostics.Looper.Attempts[0].Accepted = false

	original := record.RouteDiagnostics.Looper.Attempts[0]
	if original.Model != "" || original.Accepted == nil || !*original.Accepted {
		t.Fatalf("clone mutated original Looper diagnostics: %+v", original)
	}
}
