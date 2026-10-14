package topiccontinuity

import "math"

const (
	explicitChangeConfidence = 0.9
	disjointConfidenceMax    = 0.6
	disjointConfidenceMin    = 0.3
)

// resultInput carries everything newResult needs. Fallback is derived from
// the reason, never supplied.
type resultInput struct {
	signal     string
	class      Class
	reason     Reason
	confidence float64
	coverage   Coverage
	scope      EvidenceScope
	features   Features
}

// newResult builds every classified Result and enforces the schema v1
// invariants; a violation becomes an internal-error result.
func newResult(in resultInput) Result {
	if !validResult(in) {
		return internalErrorResult(in.signal, in.scope.AssistantIncluded)
	}
	return Result{
		SchemaVersion:    SchemaVersion,
		EvaluatorVersion: EvaluatorVersion,
		Signal:           in.signal,
		Class:            in.class,
		Confidence:       in.confidence,
		Reason:           in.reason,
		Fallback:         fallbackReason(in.reason),
		Coverage:         in.coverage,
		Scope:            in.scope,
		HistorySource:    SourceOriginalSnapshot,
		Features:         in.features,
	}
}

func validResult(in resultInput) bool {
	class, known := reasonClass[in.reason]
	if !known || class != in.class {
		return false
	}
	if in.coverage != CoverageFull && in.coverage != CoverageWindow && in.coverage != CoveragePartial {
		return false
	}
	if math.IsNaN(in.confidence) || math.IsInf(in.confidence, 0) || in.confidence < 0 || in.confidence > 1 {
		return false
	}
	switch in.class {
	case ClassUnknown:
		return in.confidence == 0
	case ClassChange:
		if in.coverage != CoverageFull {
			return false
		}
		if in.reason == ReasonExplicitChange {
			return in.confidence >= explicitChangeConfidence
		}
		return in.confidence >= disjointConfidenceMin && in.confidence <= disjointConfidenceMax
	}
	return true
}

// internalErrorResult reports an invalid rule configuration or a violated
// result invariant: partial coverage, no evidence flags, zero features.
func internalErrorResult(signal string, assistantIncluded bool) Result {
	return Result{
		SchemaVersion:    SchemaVersion,
		EvaluatorVersion: EvaluatorVersion,
		Signal:           signal,
		Class:            ClassUnknown,
		Reason:           ReasonInternalError,
		Fallback:         true,
		Coverage:         CoveragePartial,
		Scope:            EvidenceScope{AssistantIncluded: assistantIncluded},
		HistorySource:    SourceOriginalSnapshot,
	}
}
