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
	"time"
)

// FaithfulnessVerifier scores each candidate against trusted context using the
// shared hallucination-detector backend (trusted-context faithfulness: fewer
// unsupported spans => higher score). The scoring rule is identical to fusion
// grounding's context mode: 1/(1+number of unsupported spans).
type FaithfulnessVerifier struct {
	detect  HallucinationDetectFunc
	version string
}

// NewFaithfulnessVerifier wraps the hallucination-detector backend as a
// Verifier. A nil detect func reports unavailable.
func NewFaithfulnessVerifier(detect HallucinationDetectFunc) *FaithfulnessVerifier {
	return &FaithfulnessVerifier{detect: detect, version: "local/faithfulness/1"}
}

// Kind implements Verifier.
func (v *FaithfulnessVerifier) Kind() VerifierKind { return VerifierKindFaithfulness }

// Verify implements Verifier. req.TrustedContext is the source material,
// req.Task the question; each candidate is scored and its unsupported spans
// are carried in the candidate's Flags.
func (v *FaithfulnessVerifier) Verify(ctx context.Context, req *VerifierRequest) (*VerifierResult, error) {
	if v.detect == nil {
		return nil, NewVerifierError(VerifierFailureUnavailable, fmt.Errorf("hallucination detector backend not configured"))
	}
	if len(req.Candidates) == 0 {
		return nil, NewVerifierError(VerifierFailureNoCandidate, fmt.Errorf("faithfulness verifier requires a candidate"))
	}
	start := time.Now()
	best := -1.0
	scores := make([]CandidateScore, 0, len(req.Candidates))
	for _, c := range req.Candidates {
		spans, _, err := v.detect(req.TrustedContext, req.Task, c.Content)
		if err != nil {
			return nil, &VerifierError{Code: VerifierFailureUnavailable, Err: err}
		}
		// 0 unsupported spans => 1.0; degrades as spans accumulate (matches
		// fusion grounding's context mode).
		score := 1.0 / (1.0 + float64(len(spans)))
		scores = append(scores, CandidateScore{CandidateID: c.ID, Confidence: score, Flags: spans})
		if score > best {
			best = score
		}
	}
	return &VerifierResult{
		Disposition: DispositionApprove,
		Confidence:  &best,
		Kind:        v.Kind(),
		Version:     v.version,
		Scores:      scores,
		LatencyMs:   waitForLatency(start),
	}, nil
}

// PeerConsistencyVerifier scores each candidate by how well its peers entail
// (vs contradict) it, using the shared NLI backend. It is agreement evidence
// only - never a truth label, so it never approves. The scoring rule is
// identical to fusion grounding's panel mode:
// clamp01((mean(entail - penalty*contradict) + penalty) / (1 + penalty)).
type PeerConsistencyVerifier struct {
	nli     NLIClassifyFunc
	penalty float64
	version string
}

// NewPeerConsistencyVerifier wraps the NLI backend plus the contradiction
// penalty as a Verifier. A nil nli func reports unavailable.
func NewPeerConsistencyVerifier(nli NLIClassifyFunc, penalty float64) *PeerConsistencyVerifier {
	if penalty <= 0 {
		penalty = 1.0
	}
	return &PeerConsistencyVerifier{nli: nli, penalty: penalty, version: "local/peer_consistency/1"}
}

// Kind implements Verifier.
func (v *PeerConsistencyVerifier) Kind() VerifierKind { return VerifierKindPeerConsistency }

// Verify implements Verifier. Every candidate is scored against all other
// candidates (the panel as its own mutual reference); peers that contradict a
// candidate are recorded in that candidate's Flags. With fewer than two
// candidates there is no consensus evidence, so the verifier abstains (never
// tie — there is nothing to judge).
func (v *PeerConsistencyVerifier) Verify(ctx context.Context, req *VerifierRequest) (*VerifierResult, error) {
	if v.nli == nil {
		return nil, NewVerifierError(VerifierFailureUnavailable, fmt.Errorf("nli backend not configured"))
	}
	if len(req.Candidates) == 0 {
		return nil, NewVerifierError(VerifierFailureNoCandidate, fmt.Errorf("peer-consistency verifier requires a candidate"))
	}
	start := time.Now()
	best := -1.0
	scores := make([]CandidateScore, 0, len(req.Candidates))
	for i, c := range req.Candidates {
		var sum float64
		var n int
		var flags []string
		for j, peer := range req.Candidates {
			if i == j {
				continue
			}
			// Directional consistency: does peer (premise) entail/contradict
			// candidate c (hypothesis)?
			entail, contradict, err := nliPairSignalWith(v.nli, peer.Content, c.Content)
			if err != nil {
				return nil, &VerifierError{Code: VerifierFailureUnavailable, Err: err}
			}
			sum += entail - v.penalty*contradict
			n++
			if contradict > entail && contradict >= 0.5 {
				flags = append(flags, peer.ID)
			}
		}
		raw := 0.0
		if n > 0 {
			raw = sum / float64(n)
		}
		// raw is in [-penalty, 1]; map to [0,1].
		score := clamp01((raw + v.penalty) / (1.0 + v.penalty))
		scores = append(scores, CandidateScore{CandidateID: c.ID, Confidence: score, Flags: flags})
		if score > best {
			best = score
		}
	}
	r := &VerifierResult{
		Disposition: DispositionTie,
		Kind:        v.Kind(),
		Version:     v.version,
		Scores:      scores,
		LatencyMs:   waitForLatency(start),
	}
	if len(req.Candidates) < 2 {
		r.Disposition = DispositionAbstain
		r.ReasonCodes = []string{"insufficient_peers"}
		return r, nil
	}
	r.Confidence = &best
	return r, nil
}
