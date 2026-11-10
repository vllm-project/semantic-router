package config

// ScoreKind classifies what a matched signal leaf contributes to a decision's
// confidence. Ranking may only compare scores of the same kind, and a leaf
// that reports no measurement at all contributes nothing.
type ScoreKind string

const (
	// ScoreKindNone marks a policy leaf. Keyword rules, conversation and
	// metadata predicates, input modalities and projection outputs all
	// restate a decision the operator already made, so they gate eligibility
	// without adding evidence.
	ScoreKindNone ScoreKind = ""
	// ScoreKindProbability marks a model-reported probability.
	ScoreKindProbability ScoreKind = "probability"
)

// signalScoreKinds records the kind each evidence signal reports. A type is
// absent from this map when it is policy.
var signalScoreKinds = map[string]ScoreKind{
	// A systemone leaf reports the answer's probability: P(true), the chosen
	// option's probability, or a Score answer's most likely level. The
	// application signals ask through it and report the same.
	SignalTypeSystemOne:    ScoreKindProbability,
	SignalTypeDomain:       ScoreKindProbability,
	SignalTypeSafety:       ScoreKindProbability,
	SignalTypeJailbreak:    ScoreKindProbability,
	SignalTypeComplexity:   ScoreKindProbability,
	SignalTypeTurnRelation: ScoreKindProbability,
}

// SignalScoreKind reports the score kind of a signal type. Policy types report
// ScoreKindNone.
func SignalScoreKind(signalType string) ScoreKind {
	return signalScoreKinds[signalType]
}

// IsEvidenceSignal reports whether a signal type carries measured evidence.
func IsEvidenceSignal(signalType string) bool {
	return SignalScoreKind(signalType) != ScoreKindNone
}

// DeclaredScoreKinds reports the distinct score kinds a rule tree can produce.
// Selection uses it to refuse a decision whose confidence would depend on
// which branch of an OR matched.
func DeclaredScoreKinds(node *RuleNode) []ScoreKind {
	kinds := map[ScoreKind]struct{}{}
	collectScoreKinds(node, kinds)
	return sortedKinds(kinds)
}
