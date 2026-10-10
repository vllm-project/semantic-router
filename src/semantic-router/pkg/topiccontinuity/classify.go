package topiccontinuity

import "math"

// classify applies one rule's thresholds and the fixed precedence to its
// group's shared features. It does no text work.
func classify(cfg EvalConfig, prepared preparation, x extraction) Result {
	base := resultInput{signal: cfg.Name, class: ClassUnknown, coverage: x.Coverage, scope: x.Scope}
	if x.Status != "" {
		base.reason = x.Status
		return newResult(base)
	}
	base.features = x.Features
	class, reason, confidence := decide(cfg, prepared, x)
	base.class, base.reason, base.confidence = class, reason, confidence
	return newResult(base)
}

func notFull(coverage Coverage) Reason {
	if coverage == CoverageWindow {
		return ReasonHistoryBeyondWindow
	}
	return ReasonIncompleteHistory
}

// decide applies the precedence in order; the first matching case wins. Every
// change case requires full coverage and positive evidence, and ambiguity or
// missing evidence always resolves to unknown, never to change.
func decide(cfg EvalConfig, prepared preparation, x extraction) (Class, Reason, float64) {
	f := x.Features
	full := x.Coverage == CoverageFull
	s, r := f.CombinedScore, f.MaxRawScore
	c, ch := cfg.Continuation, cfg.Change
	selfContained := !f.StrongReference && !f.WeakReference && f.LiveProseTerms >= 5
	switch {
	case prepared.liveKind == liveToolContinuation:
		return ClassContinuation, ReasonToolExchange, 1.0
	case prepared.LiveOpaqueOnly:
		return ClassUnknown, ReasonOpaqueOnly, 0
	case f.ChangeMarker && (f.StrongReference || f.MaxEntityScore >= 0.5):
		return ClassUnknown, ReasonConflicting, 0
	case f.ChangeMarker && !full:
		return ClassUnknown, notFull(x.Coverage), 0
	case f.ChangeMarker:
		return ClassChange, ReasonExplicitChange, explicitChangeConfidence
	case f.StrongReference:
		return ClassContinuation, ReasonReference, 0.8
	case f.Acknowledgement:
		return ClassContinuation, ReasonAcknowledgement, 0.5
	case f.LiveTerms < 3:
		return ClassUnknown, ReasonInsufficientText, 0
	case s >= c:
		reason := ReasonLexicalOverlap
		if f.EntityScore >= f.LexicalScore {
			reason = ReasonEntityOverlap
		}
		return ClassContinuation, reason, 0.5 + 0.5*math.Min(1, (s-c)/(1-c))
	case f.MarkerAmbiguous:
		return ClassUnknown, ReasonAmbiguousMarker, 0
	case r <= ch && selfContained && !full:
		return ClassUnknown, notFull(x.Coverage), 0
	case r <= ch && selfContained:
		if ch == 0 {
			return ClassChange, ReasonDisjoint, disjointConfidenceMax
		}
		return ClassChange, ReasonDisjoint, disjointConfidenceMin + 0.3*(ch-r)/ch
	default:
		return ClassUnknown, ReasonInconclusive, 0
	}
}
