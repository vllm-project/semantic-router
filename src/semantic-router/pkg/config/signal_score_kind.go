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
	// ScoreKindUnknown marks evidence whose quantity depends on the backend
	// that produced it. A complexity rule reports a calibrated probability
	// through label_distribution.v1 and the magnitude of a prototype margin
	// on the local path, and a jailbreak rule reports a classifier
	// probability through BERT and a contrastive score under
	// `method: contrastive`. Both write the same key, so the signal type does
	// not establish the quantity and selection must not rank on it.
	ScoreKindUnknown ScoreKind = "unknown"
)

// signalScoreKinds records the kind each evidence signal reports. A type is
// absent from this map when it is policy, and maps to ScoreKindUnknown when
// the quantity depends on the configured backend.
var signalScoreKinds = map[string]ScoreKind{
	SignalTypeDomain:     ScoreKindProbability,
	SignalTypeClassifier: ScoreKindProbability,
	SignalTypeSafety:     ScoreKindProbability,
	SignalTypePreference: ScoreKindProbability,
	SignalTypeEmbedding:  ScoreKindSimilarity,
	SignalTypeReask:      ScoreKindSimilarity,
	// A KB rule reports the similarity of the matched label, not a
	// probability. See kbLabelMatchConfidence.
	SignalTypeKB:         ScoreKindSimilarity,
	SignalTypeComplexity: ScoreKindUnknown,
	SignalTypeJailbreak:  ScoreKindUnknown,
}

// SignalScoreKind reports the score kind of a signal type. Policy types report
// ScoreKindNone.
func SignalScoreKind(signalType string) ScoreKind {
	return signalScoreKinds[signalType]
}

// IsEvidenceSignal reports whether a signal type carries measured evidence. An
// evidence leaf whose kind is unknown still counts as evidence, so a decision
// that rests on it is not comparable rather than silently ranked on the rest.
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
