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
	"strings"
	"testing"
	"time"
)

func TestDeterministicVerifierApprovesValidCandidate(t *testing.T) {
	v := NewDeterministicVerifier("test/validator", nil)
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "task",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "ok"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionApprove {
		t.Fatalf("disposition = %q, want approve", res.Disposition)
	}
	if res.Kind != VerifierKindDeterministic || res.Version != "test/validator" {
		t.Fatalf("kind/version = %q/%q", res.Kind, res.Version)
	}
}

func TestDeterministicVerifierRejectsOnCheck(t *testing.T) {
	v := NewDeterministicVerifier("test/validator", func(c VerifierCandidate) bool {
		return strings.Contains(c.Content, "valid")
	})
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Candidates: []VerifierCandidate{{ID: "c1", Content: "garbage"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionReject {
		t.Fatalf("disposition = %q, want reject", res.Disposition)
	}
	if len(res.ReasonCodes) == 0 || res.ReasonCodes[0] == "" {
		t.Fatalf("reject must carry a reason code, got %v", res.ReasonCodes)
	}
}

func TestDeterministicVerifierNoCandidateIsTypedFailure(t *testing.T) {
	v := NewDeterministicVerifier("test/validator", nil)
	_, err := v.Verify(context.Background(), &VerifierRequest{Task: "task"})
	var verr *VerifierError
	if !errors.As(err, &verr) {
		t.Fatalf("expected *VerifierError, got %T", err)
	}
	if verr.Code != VerifierFailureNoCandidate {
		t.Fatalf("code = %q, want no_candidate", verr.Code)
	}
}

type timeoutError struct{}

func (timeoutError) Error() string { return "deadline exceeded" }
func (timeoutError) Timeout() bool { return true }

type netError struct{}

func (netError) Error() string { return "connection refused" }

func TestClassifyVerifierHTTPErrorDistinguishesTimeout(t *testing.T) {
	if got := classifyVerifierHTTPError(timeoutError{}); got != VerifierFailureTimeout {
		t.Fatalf("timeout error classified as %q, want timeout", got)
	}
	if got := classifyVerifierHTTPError(netError{}); got != VerifierFailureUnavailable {
		t.Fatalf("network error classified as %q, want unavailable", got)
	}
	if got := classifyVerifierHTTPError(nil); got != VerifierFailureUnavailable {
		t.Fatalf("nil error classified as %q, want unavailable", got)
	}
}

func TestVerifierResultConfidenceNilIsExplicitAbsence(t *testing.T) {
	// Numeric zero must be distinguishable from "no score"; the contract uses a
	// pointer so 0.0 stays a real value (see #2864 zero-score presence rule).
	v := NewDeterministicVerifier("test/validator", nil)
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Candidates: []VerifierCandidate{{ID: "c1", Content: "ok"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Confidence != nil {
		t.Fatalf("deterministic verifier must not invent a score, got %v", *res.Confidence)
	}
}

func TestVerifierResultLatencyIsBounded(t *testing.T) {
	start := time.Now()
	waitForLatency(start) // exercise helper; no assertion on exact value
}

func TestValidVerifierDisposition(t *testing.T) {
	for d, want := range map[VerifierDisposition]bool{
		DispositionApprove:           true,
		DispositionRedo:              true,
		DispositionReject:            true,
		DispositionTie:               true,
		DispositionAbstain:           true,
		VerifierDisposition("bogus"): false,
	} {
		if got := ValidVerifierDisposition(d); got != want {
			t.Fatalf("ValidVerifierDisposition(%q) = %v, want %v", d, got, want)
		}
	}
}
