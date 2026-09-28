package tasks

import "errors"

// ErrProbabilitiesUnavailable means the provider selected a label without
// reporting its probability. A categorical verdict must never be converted to
// a synthetic softmax distribution or used as a probability threshold.
var ErrProbabilitiesUnavailable = errors.New("model does not report probabilities")

// LabelDecision preserves a categorical model answer. SourceLabel retains an
// adapter's original verdict when Label is mapped into a configured label set.
// Optional scores have their actual semantics and are not implied by the label.
type LabelDecision struct {
	Label          string          `json:"label"`
	SourceLabel    string          `json:"source_label,omitempty"`
	Categories     []string        `json:"categories,omitempty"`
	Score          *ScoreResult    `json:"score,omitempty"`
	ScoreSemantics *ScoreSemantics `json:"score_semantics,omitempty"`
}
