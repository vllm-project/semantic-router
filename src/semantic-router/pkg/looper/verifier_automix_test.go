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
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// serveAutoMixVerifier returns a handler resolving the AutoMix /verify wire,
// and the request the server last saw (question/answer/threshold).
func serveAutoMixVerifier(t *testing.T, respond func(req map[string]interface{}) (int, interface{})) (
	*httptest.Server, func() map[string]interface{},
) {
	t.Helper()
	var last map[string]interface{}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&last)
		code, payload := respond(last)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		if payload != nil {
			_ = json.NewEncoder(w).Encode(payload)
		}
	})
	srv := httptest.NewServer(handler)
	return srv, func() map[string]interface{} { return last }
}

func automixVerifierBody(code int, confidence float64) func(map[string]interface{}) (int, interface{}) {
	return func(_ map[string]interface{}) (int, interface{}) {
		return code, selection.AutoMixVerifyResponse{
			Confidence:     confidence,
			ShouldEscalate: confidence < 0.7,
			VerifiedCount:  1,
			TotalSamples:   1,
			Threshold:      0.7,
		}
	}
}

func TestAutoMixVerifierApprovesAtOrAboveThreshold(t *testing.T) {
	srv, _ := serveAutoMixVerifier(t, automixVerifierBody(200, 0.9))
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 5, 0, 0.7)
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "what is 2+2?",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "4"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionApprove || res.Confidence == nil || *res.Confidence != 0.9 {
		t.Fatalf("got disposition=%q confidence=%v", res.Disposition, res.Confidence)
	}
	if res.Kind != VerifierKindFaithfulness {
		t.Fatalf("kind = %q, want faithfulness", res.Kind)
	}
}

func TestAutoMixVerifierRedoBelowThreshold(t *testing.T) {
	srv, _ := serveAutoMixVerifier(t, automixVerifierBody(200, 0.4))
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 5, 0, 0.7)
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "q",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "a"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionRedo {
		t.Fatalf("disposition = %q, want redo", res.Disposition)
	}
}

func TestAutoMixVerifierPassesTaskAndThresholdToServer(t *testing.T) {
	srv, last := serveAutoMixVerifier(t, automixVerifierBody(200, 0.8))
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 5, 0, 0.7)
	_, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "original question",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "answer text"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := last()
	if got["question"] != "original question" || got["answer"] != "answer text" {
		t.Fatalf("wire payload = %v", got)
	}
	if got["threshold"].(float64) != 0.7 {
		t.Fatalf("threshold not forwarded: %v", got["threshold"])
	}
	if _, present := got["context"]; present {
		t.Fatalf("empty context must be omitted from the wire payload")
	}
}

func TestAutoMixVerifierScoresEveryCandidate(t *testing.T) {
	reqs := 0
	srv, _ := serveAutoMixVerifier(t, func(_ map[string]interface{}) (int, interface{}) {
		reqs++
		return 200, selection.AutoMixVerifyResponse{
			Confidence:    float64(reqs) / 10.0, // 0.1 then 0.2
			VerifiedCount: 1, TotalSamples: 1, Threshold: 0.7,
		}
	})
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 5, 0, 0.7)
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Task: "q",
		Candidates: []VerifierCandidate{
			{ID: "c1", Content: "a"},
			{ID: "c2", Content: "b"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Scores) != 2 || res.Scores[1].Confidence != 0.2 {
		t.Fatalf("scores = %+v, want per-candidate 0.1/0.2", res.Scores)
	}
	if res.Confidence == nil || *res.Confidence != 0.2 {
		t.Fatalf("top-level confidence = %v, want 0.2 (best)", res.Confidence)
	}
}

func TestAutoMixVerifierTimeoutIsTyped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(3 * time.Second)
	}))
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 1, 0, 0.7)
	start := time.Now()
	_, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "q",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "a"}},
	})
	if elapsed := time.Since(start); elapsed > 2500*time.Millisecond {
		t.Fatalf("timeout did not bound the call: %s", elapsed)
	}
	var verr *VerifierError
	if !errors.As(err, &verr) {
		t.Fatalf("expected *VerifierError, got %T", err)
	}
	if verr.Code != VerifierFailureTimeout {
		t.Fatalf("failure code = %q, want timeout", verr.Code)
	}
}

func TestAutoMixVerifierMalformedOutputIsTyped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{not-json`))
	}))
	defer srv.Close()
	v := NewAutoMixVerifier(srv.URL, 5, 0, 0.7)
	_, err := v.Verify(context.Background(), &VerifierRequest{
		Task:       "q",
		Candidates: []VerifierCandidate{{ID: "c1", Content: "a"}},
	})
	var verr *VerifierError
	if !errors.As(err, &verr) {
		t.Fatalf("expected *VerifierError, got %T", err)
	}
	if verr.Code != VerifierFailureMalformed {
		t.Fatalf("failure code = %q, want malformed_output", verr.Code)
	}
}

func TestAutoMixVerifierNoCandidateIsTyped(t *testing.T) {
	v := NewAutoMixVerifier("http://127.0.0.1:1", 1, 0, 0.7)
	_, err := v.Verify(context.Background(), &VerifierRequest{Task: "q"})
	var verr *VerifierError
	if !errors.As(err, &verr) {
		t.Fatalf("expected *VerifierError, got %T", err)
	}
	if verr.Code != VerifierFailureNoCandidate {
		t.Fatalf("failure code = %q, want no_candidate", verr.Code)
	}
}
