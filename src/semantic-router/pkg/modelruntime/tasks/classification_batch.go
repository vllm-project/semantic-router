package tasks

// ClassificationBatch contains separate per-input results from the three
// maintained classification tasks. It does not describe a joint model forward.
type ClassificationBatch struct {
	Intent   []LabelDistribution
	PII      []TokenClassificationResult
	Security []LabelDistribution
}
