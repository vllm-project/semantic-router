package config

// ScoreKind classifies what a matched signal leaf contributes to a decision's
// confidence. Ranking may only compare scores of the same kind, and a leaf
// that reports no measurement at all contributes nothing.
type ScoreKind string

const (
	// ScoreKindNone marks a policy leaf. Keyword rules, conversation and
	// metadata predicates, projection outputs, structural and event rules all
	// restate a decision the operator already made, so they gate eligibility
	// without adding evidence.
	ScoreKindNone ScoreKind = ""
	// ScoreKindProbability marks a model-reported probability.
	ScoreKindProbability ScoreKind = "probability"
	// ScoreKindSimilarity marks a vector similarity.
	ScoreKindSimilarity ScoreKind = "similarity"
)

// signalScoreKinds records the kind each signal type reports. Only the types
// listed here write a measurement into SignalConfidences that came from a
// model: a category probability, a classifier score, or a similarity. Every
// other type is policy and is absent from this table.
var signalScoreKinds = map[string]ScoreKind{
	SignalTypeDomain:     ScoreKindProbability,
	SignalTypeClassifier: ScoreKindProbability,
	SignalTypeComplexity: ScoreKindProbability,
	SignalTypeJailbreak:  ScoreKindProbability,
	SignalTypeSafety:     ScoreKindProbability,
	SignalTypePreference: ScoreKindProbability,
	SignalTypeKB:         ScoreKindProbability,
	SignalTypeEmbedding:  ScoreKindSimilarity,
	SignalTypeReask:      ScoreKindSimilarity,
}

// SignalScoreKind reports the score kind of a signal type. Policy leaves and
// unknown types report ScoreKindNone.
func SignalScoreKind(signalType string) ScoreKind {
	return signalScoreKinds[signalType]
}
