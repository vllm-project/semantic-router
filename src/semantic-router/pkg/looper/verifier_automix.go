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
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// AutoMixVerifier is the shared-contract adapter for the AutoMix
// self-verification server (arXiv:2310.12963 §3.2), wrapped behind the HTTP
// client already used by confidence AutoMix entailment. It classifies as
// trusted-context faithfulness: each candidate answer is checked for
// entailment against the task (and optional trusted context). Disposition
// mirrors the existing confidence path's accept rule
// (confidence >= threshold) so public behavior is unchanged.
type AutoMixVerifier struct {
	client    *selection.AutoMixVerifierClient
	threshold float64
	version   string
}

// NewAutoMixVerifier builds an adapter around the shared cached AutoMix HTTP
// client. threshold reproduces the caller's accept rule; a non-positive
// timeout uses the client default, matching the existing confidence path.
func NewAutoMixVerifier(serverURL string, timeoutSeconds int, maxResponseBytes int64, threshold float64) *AutoMixVerifier {
	return &AutoMixVerifier{
		client:    getAutoMixVerifierClient(serverURL, timeoutSeconds, maxResponseBytes),
		threshold: threshold,
		version:   "automix/1",
	}
}

// Kind implements Verifier.
func (v *AutoMixVerifier) Kind() VerifierKind { return VerifierKindFaithfulness }

// Verify implements Verifier. A single candidate yields the top-level
// disposition/confidence exactly as the confidence AutoMix path computes it;
// multiple candidates are scored independently, the top-level confidence is
// the best score, and disposition approves iff the best candidate clears the
// threshold (rerank-friendly).
func (v *AutoMixVerifier) Verify(ctx context.Context, req *VerifierRequest) (*VerifierResult, error) {
	if len(req.Candidates) == 0 {
		return nil, NewVerifierError(VerifierFailureNoCandidate, fmt.Errorf("automix verifier requires a candidate"))
	}
	start := time.Now()
	best := -1.0
	bestIdx := -1
	scores := make([]CandidateScore, 0, len(req.Candidates))
	for i, c := range req.Candidates {
		resp, err := v.client.Verify(ctx, req.Task, c.Content, req.TrustedContext, v.threshold)
		if err != nil {
			return nil, &VerifierError{Code: classifyVerifierOutputError(err), Err: err}
		}
		scores = append(scores, CandidateScore{CandidateID: c.ID, Confidence: resp.Confidence})
		if resp.Confidence > best {
			best, bestIdx = resp.Confidence, i
		}
	}
	r := &VerifierResult{
		Confidence: &best,
		Kind:       v.Kind(),
		Version:    v.version,
		Scores:     scores,
		LatencyMs:  waitForLatency(start),
	}
	if bestIdx < 0 || best < v.threshold {
		r.Disposition = DispositionRedo
		r.ReasonCodes = []string{"below_threshold"}
		return r, nil
	}
	r.Disposition = DispositionApprove
	r.ReasonCodes = []string{"above_threshold"}
	return r, nil
}

// classifyVerifierOutputError maps an adapter client error to a typed
// VerifierFailureCode. The AutoMix client currently reports decode failures as
// plain errors; we key off its stable in-repo message to distinguish
// malformed output from transport availability. # ponytail: string match on
// the selection client's error text; replace with a typed sentinel when that
// client grows one.
func classifyVerifierOutputError(err error) VerifierFailureCode {
	if strings.Contains(err.Error(), "failed to decode response") {
		return VerifierFailureMalformed
	}
	return classifyVerifierHTTPError(err)
}
