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
	"math"
	"strings"
	"testing"
)

func TestFaithfulnessVerifierScoresBySpanCount(t *testing.T) {
	v := NewFaithfulnessVerifier(func(_ context.Context, contextText, question, answer string) ([]string, float32, error) {
		if strings.Contains(answer, "bad") {
			return []string{"unsupported-1", "unsupported-2"}, 0.4, nil
		}
		return nil, 0.9, nil
	})
	res, err := v.Verify(context.Background(), &VerifierRequest{
		Task:           "q",
		TrustedContext: "ctx",
		Candidates: []VerifierCandidate{
			{ID: "p0", Content: "good answer"},
			{ID: "p1", Content: "bad answer with spans"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionTie || res.Kind != VerifierKindFaithfulness {
		t.Fatalf("disposition/kind = %q/%q, want %q/%q (faithfulness is evidence, never a decision)",
			res.Disposition, res.Kind, DispositionTie, VerifierKindFaithfulness)
	}
	if len(res.Scores) != 2 {
		t.Fatalf("len(scores) = %d, want 2", len(res.Scores))
	}
	if res.Scores[0].Confidence != 1.0 || res.Scores[0].Flags != nil {
		t.Fatalf("good candidate = %+v, want 1.0/no flags", res.Scores[0])
	}
	if res.Scores[1].Confidence != 1.0/3.0 {
		t.Fatalf("flagged candidate confidence = %v, want %v (1/(1+2 spans))", res.Scores[1].Confidence, 1.0/3.0)
	}
	if len(res.Scores[1].Flags) != 2 {
		t.Fatalf("flagged candidate spans = %v, want 2", res.Scores[1].Flags)
	}
}

func TestFaithfulnessVerifierNilBackendIsTyped(t *testing.T) {
	v := NewFaithfulnessVerifier(nil)
	_, err := v.Verify(context.Background(), &VerifierRequest{
		Candidates: []VerifierCandidate{{ID: "p0", Content: "a"}},
	})
	var verr *VerifierError
	if !errors.As(err, &verr) || verr.Code != VerifierFailureUnavailable {
		t.Fatalf("err = %v, want typed unavailable", err)
	}
}

func TestPeerConsistencyVerifierMeanScoringAndFlags(t *testing.T) {
	// Read against a peer, an answer containing "bad" has a span at 0.8 and a
	// "partly" answer one at 0.3; anything else is fully supported.
	detect := func(_ context.Context, _, _, answer string) ([]string, float32, error) {
		switch {
		case strings.Contains(answer, "bad"):
			return []string{"bad"}, 0.8, nil
		case strings.Contains(answer, "partly"):
			return []string{"partly"}, 0.3, nil
		}
		return nil, 0, nil
	}
	v := NewPeerConsistencyVerifier(detect, 1.0)
	res, err := v.Verify(context.Background(), &VerifierRequest{Task: "q", Candidates: []VerifierCandidate{
		{ID: "p0", Content: "good one"},
		{ID: "p1", Content: "partly two"},
		{ID: "p2", Content: "bad three"},
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionTie || res.Kind != VerifierKindPeerConsistency {
		t.Fatalf("disposition/kind = %q/%q, want tie/peer_consistency", res.Disposition, res.Kind)
	}
	// support - penalty * contradiction, mapped from [-1, 1] to [0, 1]:
	// supported (1 + 1) / 2, partly (0.4 + 1) / 2, contradicted (-0.6 + 1) / 2.
	for i, want := range []float64{1.0, 0.7, 0.2} {
		if math.Abs(res.Scores[i].Confidence-want) > 1e-6 {
			t.Fatalf("p%d score = %v, want %v", i, res.Scores[i].Confidence, want)
		}
	}
	if res.Scores[0].Flags != nil || res.Scores[1].Flags != nil {
		t.Fatalf("flags = %v / %v, want none below a 0.5 contradiction", res.Scores[0].Flags, res.Scores[1].Flags)
	}
	if len(res.Scores[2].Flags) != 2 || res.Scores[2].Flags[0] != "p0" || res.Scores[2].Flags[1] != "p1" {
		t.Fatalf("p2 flags = %v, want [p0 p1] (contradicted by both peers)", res.Scores[2].Flags)
	}
}

func TestPeerConsistencyVerifierPenaltyWeighsContradiction(t *testing.T) {
	detect := func(_ context.Context, _, _, answer string) ([]string, float32, error) {
		if answer == "bad" {
			return []string{"bad"}, 0.5, nil
		}
		return nil, 0, nil
	}
	res, err := NewPeerConsistencyVerifier(detect, 3.0).Verify(context.Background(), &VerifierRequest{Candidates: []VerifierCandidate{
		{ID: "p0", Content: "good"},
		{ID: "p1", Content: "bad"},
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// (0.5 - 3 * 0.5 + 3) / (1 + 3).
	if math.Abs(res.Scores[1].Confidence-0.5) > 1e-6 {
		t.Fatalf("p1 score = %v, want 0.5", res.Scores[1].Confidence)
	}
}

func TestPeerConsistencyVerifierAbstainsWithoutPeers(t *testing.T) {
	detect := func(_ context.Context, _, _, _ string) ([]string, float32, error) { return nil, 0, nil }
	v := NewPeerConsistencyVerifier(detect, 1.0)
	res, err := v.Verify(context.Background(), &VerifierRequest{Candidates: []VerifierCandidate{
		{ID: "p0", Content: "only one"},
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Disposition != DispositionAbstain {
		t.Fatalf("disposition = %q, want abstain (no consensus evidence)", res.Disposition)
	}
	if res.Confidence != nil {
		t.Fatalf("abstain must not invent a confidence, got %v", *res.Confidence)
	}
	// Scores stay populated for downstream ranking.
	if len(res.Scores) != 1 {
		t.Fatalf("scores = %v", res.Scores)
	}
}

func TestPeerConsistencyVerifierDetectorErrorIsTyped(t *testing.T) {
	detect := func(_ context.Context, _, _, _ string) ([]string, float32, error) {
		return nil, 0, errors.New("runtime unavailable")
	}
	_, err := NewPeerConsistencyVerifier(detect, 1.0).Verify(context.Background(), &VerifierRequest{
		Candidates: []VerifierCandidate{{ID: "p0", Content: "a"}, {ID: "p1", Content: "b"}},
	})
	var verr *VerifierError
	if !errors.As(err, &verr) || verr.Code != VerifierFailureUnavailable {
		t.Fatalf("err = %v, want typed unavailable", err)
	}
}

func TestPeerConsistencyVerifierNilBackendIsTyped(t *testing.T) {
	v := NewPeerConsistencyVerifier(nil, 1.0)
	_, err := v.Verify(context.Background(), &VerifierRequest{
		Candidates: []VerifierCandidate{{ID: "p0", Content: "a"}, {ID: "p1", Content: "b"}},
	})
	var verr *VerifierError
	if !errors.As(err, &verr) || verr.Code != VerifierFailureUnavailable {
		t.Fatalf("err = %v, want typed unavailable", err)
	}
}

func TestGroundingScoresFromVerifierAlignsNilSlots(t *testing.T) {
	// A panel with a nil slot in the middle must keep index alignment.
	panel := []*ModelResponse{{Model: "a", Content: "aa"}, nil, {Model: "b", Content: "bb"}}
	candidates, idx := groundingVerifierCandidates(panel)
	if len(candidates) != 2 || idx[0] != 0 || idx[1] != 2 {
		t.Fatalf("candidates/idx = %v/%v", candidates, idx)
	}
	res := &VerifierResult{
		Disposition: DispositionTie,
		Kind:        VerifierKindPeerConsistency,
		Scores: []CandidateScore{
			{CandidateID: candidates[0].ID, Confidence: 0.8, Flags: []string{"peer"}},
			{CandidateID: candidates[1].ID, Confidence: 0.2},
		},
	}
	scores := groundingScoresFromVerifier(res, panel, idx)
	if len(scores) != 3 {
		t.Fatalf("len(scores) = %d, want 3 (index alignment)", len(scores))
	}
	for _, tc := range []struct {
		i        int
		model    string
		score    float64
		hasSpans bool
	}{
		{0, "a", 0.8, true},
		{1, "", 0, false},
		{2, "b", 0.2, false},
	} {
		if scores[tc.i].Model != tc.model || scores[tc.i].Score != tc.score ||
			(scores[tc.i].FlaggedSpans != nil) != tc.hasSpans {
			t.Fatalf("scores[%d] = %+v", tc.i, scores[tc.i])
		}
	}
}

func TestGroundingScoresPreserveLiteralFaithfulnessSpans(t *testing.T) {
	// Regression (#2857): faithfulness flags are arbitrary unsupported text
	// spans. A span that literally equals an internal candidate ID must pass
	// through verbatim; only peer-consistency flags are ID-remapped.
	panel := []*ModelResponse{{Model: "a", Content: "aa"}}
	candidates, idx := groundingVerifierCandidates(panel)
	res := &VerifierResult{
		Disposition: DispositionApprove,
		Kind:        VerifierKindFaithfulness,
		Scores: []CandidateScore{
			{CandidateID: candidates[0].ID, Confidence: 0.5, Flags: []string{"panel-0", "unsupported claim"}},
		},
	}
	scores := groundingScoresFromVerifier(res, panel, idx)
	if len(scores) != 1 {
		t.Fatalf("scores = %+v", scores)
	}
	got := scores[0].FlaggedSpans
	if len(got) != 2 || got[0] != "panel-0" || got[1] != "unsupported claim" {
		t.Fatalf("faithfulness spans must pass through verbatim, got %v", got)
	}
}
