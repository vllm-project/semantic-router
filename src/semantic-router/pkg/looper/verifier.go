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
	"fmt"
	"time"
)

// VerifierKind distinguishes verifier semantics instead of flattening every
// signal into a single "confidence" score. The taxonomy mirrors the MoM &
// Routing workgroup's verifier contract (issue #2857): each kind carries a
// distinct interpretation, so policy can treat a deterministic check
// differently from a model judge or peer-agreement evidence.
type VerifierKind string

const (
	// VerifierKindDeterministic verifies by tests, parser/schema checks, SQL
	// execution, or an exact/reference answer. Ground truth by construction.
	VerifierKindDeterministic VerifierKind = "deterministic_outcome"

	// VerifierKindFaithfulness checks candidate answers against trusted
	// context (RAG/tool sources) for entailment/hallucination. Measures
	// faithfulness, never truth.
	VerifierKindFaithfulness VerifierKind = "trusted_context_faithfulness"

	// VerifierKindOutcomeReward scores a whole candidate with a scalar reward.
	VerifierKindOutcomeReward VerifierKind = "outcome_reward_model"

	// VerifierKindProcessReward scores partial steps; only meaningful when the
	// backend/search can act on prefixes.
	VerifierKindProcessReward VerifierKind = "process_reward_model"

	// VerifierKindLLMJudge is an LLM scalar/rerank judge for outcome reranking.
	VerifierKindLLMJudge VerifierKind = "llm_judge"

	// VerifierKindPairwise compares candidate pairs for tournament/league
	// selection.
	VerifierKindPairwise VerifierKind = "pairwise_comparator"

	// VerifierKindPeerConsistency is agreement evidence only; it is never a
	// truth label.
	VerifierKindPeerConsistency VerifierKind = "peer_consistency"
)

// VerifierDisposition is the typed outcome a verifier returns. It is
// deliberately coarse so policy stays deterministic: disposition drives the
// decision, confidence explains it, reason codes and evidence bound it.
type VerifierDisposition string

const (
	// DispositionApprove accepts the candidate as policy-valid.
	DispositionApprove VerifierDisposition = "approve"
	// DispositionRedo rejects this attempt but permits escalation/retry.
	DispositionRedo VerifierDisposition = "redo"
	// DispositionReject removes the candidate outright.
	DispositionReject VerifierDisposition = "reject"
	// DispositionTie reports ambiguity; the caller's tie policy decides.
	DispositionTie VerifierDisposition = "tie"
	// DispositionAbstain reports the verifier cannot judge (not an error).
	DispositionAbstain VerifierDisposition = "abstain"
)

// ValidVerifierDisposition reports whether d is a known disposition.
func ValidVerifierDisposition(d VerifierDisposition) bool {
	switch d {
	case DispositionApprove, DispositionRedo, DispositionReject, DispositionTie, DispositionAbstain:
		return true
	default:
		return false
	}
}

// VerifierCandidate is one outcome presented to a verifier. CandidateID is
// opaque and stable for the verification; model-based judges must NOT see the
// producing model/identity beyond this ID (the caller owns the blind mapping
// and its Replay record).
type VerifierCandidate struct {
	// ID is an opaque, stable identifier for this candidate.
	ID string
	// Content is the candidate response text (or whatever the verifier
	// consumes: prompt-role content, tool span, etc.).
	Content string
}

// VerifierRequest is the normalized input envelope for verification. The task
// and trusted context are passed separately from candidate text so
// faithful/hallucination verifiers can distinguish the question/source from
// the answer — never concatenate them before the verifier.
type VerifierRequest struct {
	// Task is the original user request/task text.
	Task string
	// Candidates are the normalized candidate outcomes to verify.
	Candidates []VerifierCandidate
	// TrustedContext is optional RAG/tool source material. Only
	// faithfulness-kind verifiers consume it; it must never be treated as
	// peer agreement.
	TrustedContext string
}

// CandidateScore is an optional per-candidate score for reranking judges and
// peer/faithfulness scoring over multiple candidates. It keeps the contract
// rerank-capable without a second algorithm-specific interface; single-candidate
// verifiers leave it empty and use the top-level disposition/confidence. Flags
// carry bounded per-candidate evidence (unsupported spans, contradicting peer
// IDs) for audit and downstream presentation.
type CandidateScore struct {
	CandidateID string   `json:"candidate_id"`
	Confidence  float64  `json:"confidence"`
	Flags       []string `json:"flags,omitempty"`
}

// VerifierResult is the typed outcome of one verification. Confidence is a
// pointer so numeric zero stays a valid value and nil explicitly means "no
// score" (abstain or deterministic verdicts). All fields are stable enough
// for Replay/audit.
type VerifierResult struct {
	// Disposition is the typed decision.
	Disposition VerifierDisposition `json:"disposition"`
	// Confidence is the bounded verification score in [0,1] when the
	// verifier produces one; nil when unset.
	Confidence *float64 `json:"confidence,omitempty"`
	// ReasonCodes are stable, low-cardinality reasons (e.g. "below_threshold",
	// "malformed", "abstain").
	ReasonCodes []string `json:"reason_codes,omitempty"`
	// Kind is the verifier semantics that produced this result.
	Kind VerifierKind `json:"kind"`
	// Version identifies the verifier implementation/wire version for audit.
	Version string `json:"version,omitempty"`
	// Scores carries per-candidate confidence when the request had multiple
	// candidates and the verifier can score them independently.
	Scores []CandidateScore `json:"scores,omitempty"`
	// Usage accounts verifier compute (external model calls, etc.) so it can
	// be folded into attempt/budget accounting.
	Usage TokenUsage `json:"usage,omitempty"`
	// LatencyMs is the verifier wall-clock time.
	LatencyMs int64 `json:"latency_ms,omitempty"`
}

// Verifier is the shared outcome-verification contract consumed by Looper
// algorithm paths (confidence AutoMix, Fusion grounding, future Best-of-N /
// reranking) so no algorithm embeds its own verifier protocol.
type Verifier interface {
	// Kind reports the verifier semantics.
	Kind() VerifierKind
	// Verify verifies the normalized candidates. It returns a typed
	// VerifierResult or a *VerifierError with a stable failure code; policy
	// maps the failure code to its on_error behavior.
	Verify(ctx context.Context, req *VerifierRequest) (*VerifierResult, error)
}

// VerifierFailureCode classifies typed verifier failures so on_error
// semantics stay deterministic across adapters.
type VerifierFailureCode string

const (
	// VerifierFailureTimeout means the verifier exceeded its bound.
	VerifierFailureTimeout VerifierFailureCode = "timeout"
	// VerifierFailureUnavailable means the verifier could not be reached or
	// was not configured/ready.
	VerifierFailureUnavailable VerifierFailureCode = "unavailable"
	// VerifierFailureMalformed means the verifier returned unparsable output.
	VerifierFailureMalformed VerifierFailureCode = "malformed_output"
	// VerifierFailureNoCandidate means no policy-valid candidate was supplied.
	VerifierFailureNoCandidate VerifierFailureCode = "no_candidate"
)

// VerifierError is the typed verifier failure. Callers switch on Code for
// on_error determinism; the wrapped error carries adapter detail for logs.
type VerifierError struct {
	Code VerifierFailureCode
	Err  error
}

func (e *VerifierError) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("verifier %s", e.Code)
	}
	return fmt.Sprintf("verifier %s: %v", e.Code, e.Err)
}

func (e *VerifierError) Unwrap() error { return e.Err }

// NewVerifierError returns a typed verifier failure using the io-style %w
// convention inherited by callers.
func NewVerifierError(code VerifierFailureCode, err error) *VerifierError {
	return &VerifierError{Code: code, Err: err}
}

// classifyVerifierHTTPError maps a client-side HTTP error to a typed failure
// code. Adapters that build on net/http reuse this so timeout-vs-unavailable
// semantics stay uniform.
func classifyVerifierHTTPError(err error) VerifierFailureCode {
	var nerr interface{ Timeout() bool }
	if errors.As(err, &nerr) && nerr.Timeout() {
		return VerifierFailureTimeout
	}
	return VerifierFailureUnavailable
}

// waitForLatency measures caller-side latency into the result when the
// adapter does not already carry it. Kept tiny; adapters may set LatencyMs
// directly.
func waitForLatency(start time.Time) int64 {
	return time.Since(start).Milliseconds()
}

// DeterministicVerifier is a pluggable deterministic check used by tests and
// by recipes that want an exact/parser-style outcome gate. It approves every
// candidate unless a configured rejector returns false; abstains when no
// candidate is supplied. It is the reference contract implementation.
type DeterministicVerifier struct {
	// Name is reflected in Version for audit.
	Name string
	// Accept reports whether the candidate content is policy-valid. Nil
	// accepts everything.
	Accept func(candidate VerifierCandidate) bool
}

// NewDeterministicVerifier returns a deterministic verifier; when accept is
// nil every candidate is approved.
func NewDeterministicVerifier(name string, accept func(candidate VerifierCandidate) bool) *DeterministicVerifier {
	return &DeterministicVerifier{Name: name, Accept: accept}
}

// Kind implements Verifier.
func (v *DeterministicVerifier) Kind() VerifierKind { return VerifierKindDeterministic }

// Verify implements Verifier with deterministic disposition semantics.
func (v *DeterministicVerifier) Verify(ctx context.Context, req *VerifierRequest) (*VerifierResult, error) {
	if ctx == nil {
		return nil, NewVerifierError(VerifierFailureUnavailable, fmt.Errorf("nil context"))
	}
	if len(req.Candidates) == 0 {
		return nil, NewVerifierError(VerifierFailureNoCandidate, fmt.Errorf("deterministic verifier requires a candidate"))
	}
	accept := v.Accept
	if accept == nil {
		accept = func(VerifierCandidate) bool { return true }
	}
	for _, c := range req.Candidates {
		if !accept(c) {
			return &VerifierResult{
				Disposition: DispositionReject,
				ReasonCodes: []string{"deterministic_failed"},
				Kind:        v.Kind(),
				Version:     v.Name,
			}, nil
		}
	}
	return &VerifierResult{
		Disposition: DispositionApprove,
		ReasonCodes: []string{"deterministic_passed"},
		Kind:        v.Kind(),
		Version:     v.Name,
	}, nil
}
