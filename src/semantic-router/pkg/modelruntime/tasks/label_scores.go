package tasks

import (
	"fmt"
	"math"
)

// LabelScores contains independent per-label probabilities in the prepared
// binding's label order. Unlike LabelDistribution, their sum is unrestricted.
type LabelScores struct {
	Scores []float32
	Input  *InputUsage
}

func (r LabelScores) InputMetadata() *InputUsage { return r.Input }

// TextWindowsRequest requests complete token coverage from a single owned
// tokenizer. Size includes special tokens; Overlap counts content tokens.
type TextWindowsRequest struct {
	Text    string
	Size    int
	Overlap int
}

type LabelDistributionWindow struct {
	Start, End    int // Half-open offsets in the original content token IDs.
	Probabilities []float32
}

type LabelScoresWindow struct {
	Start, End int
	Scores     []float32
}

type WindowedLabelDistribution struct {
	Windows       []LabelDistributionWindow
	ContentTokens int
	Input         *InputUsage
}

type WindowedLabelScores struct {
	Windows       []LabelScoresWindow
	ContentTokens int
	Input         *InputUsage
}

func (r WindowedLabelDistribution) InputMetadata() *InputUsage { return r.Input }
func (r WindowedLabelScores) InputMetadata() *InputUsage       { return r.Input }

// ValidateLabelScores rejects malformed probabilities without renormalizing
// them. Independent sigmoid heads must never pass through a softmax contract.
func ValidateLabelScores(scores []float32) error {
	if len(scores) == 0 {
		return fmt.Errorf("label scores are empty")
	}
	for _, score := range scores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score < 0 || score > 1 {
			return fmt.Errorf("label score is outside [0,1]")
		}
	}
	return nil
}

func ValidateTextWindows(input TextWindowsRequest) error {
	if input.Size <= 0 || input.Overlap < 0 || input.Overlap >= input.Size {
		return fmt.Errorf("window size must be positive and overlap smaller than size")
	}
	return nil
}

// ValidateWindowCoverage checks offsets without estimating tokens from text.
// Native tokenization supplies ContentTokens; empty content has one [0,0) window.
func ValidateWindowCoverage(contentTokens int, spans [][2]int, input *InputUsage) error {
	if contentTokens < 0 || len(spans) == 0 || input == nil || input.Truncated || input.OriginalTokens != input.ProcessedTokens || input.OriginalTokens < contentTokens {
		return fmt.Errorf("window result does not declare complete input coverage")
	}
	start, end := 0, 0
	for i, span := range spans {
		if span[0] < 0 || span[1] < span[0] || span[1] > contentTokens || (i == 0 && span[0] != 0) || span[0] > end || (i > 0 && (span[0] <= start || span[1] <= end)) {
			return fmt.Errorf("window token offsets are invalid or leave a gap")
		}
		start, end = span[0], span[1]
	}
	if end != contentTokens {
		return fmt.Errorf("windows do not cover the final content token")
	}
	return nil
}
