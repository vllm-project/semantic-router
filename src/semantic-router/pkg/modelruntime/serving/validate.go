package serving

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func validateText(text string) error {
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("model input text must not be empty")
	}
	return nil
}

func validateWindowInput(input tasks.TextWindowsRequest) error {
	if err := validateText(input.Text); err != nil {
		return err
	}
	return tasks.ValidateTextWindows(input)
}

func validateGroundedInput(input tasks.GroundedTextRequest) error {
	if err := validateText(input.Context); err != nil {
		return err
	}
	return validateText(input.Answer)
}

func validateDistributionResult(_ string, result tasks.LabelDistribution) error {
	return validateDistribution(result.Probabilities)
}

func validateDistribution(probabilities []float32) error {
	if len(probabilities) == 0 {
		return fmt.Errorf("label distribution is empty")
	}
	var sum float64
	for _, value := range probabilities {
		p := float64(value)
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return fmt.Errorf("label probability is outside [0,1]")
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-3 {
		return fmt.Errorf("label probabilities do not sum to one")
	}
	return nil
}

func validateScoresResult(_ string, result tasks.LabelScores) error {
	return tasks.ValidateLabelScores(result.Scores)
}

func validateWindowDistribution(_ tasks.TextWindowsRequest, result tasks.WindowedLabelDistribution) error {
	spans := make([][2]int, len(result.Windows))
	for i, window := range result.Windows {
		if len(window.Probabilities) != len(result.Windows[0].Probabilities) {
			return fmt.Errorf("window class counts differ")
		}
		if err := validateDistribution(window.Probabilities); err != nil {
			return err
		}
		spans[i] = [2]int{window.Start, window.End}
	}
	return tasks.ValidateWindowCoverage(result.ContentTokens, spans, result.Input)
}

func validateWindowScores(_ tasks.TextWindowsRequest, result tasks.WindowedLabelScores) error {
	spans := make([][2]int, len(result.Windows))
	for i, window := range result.Windows {
		if len(window.Scores) != len(result.Windows[0].Scores) {
			return fmt.Errorf("window label counts differ")
		}
		if err := tasks.ValidateLabelScores(window.Scores); err != nil {
			return err
		}
		spans[i] = [2]int{window.Start, window.End}
	}
	return tasks.ValidateWindowCoverage(result.ContentTokens, spans, result.Input)
}

func validateWindowTokens(input tasks.TextWindowsRequest, output tasks.WindowedTokenClassification) error {
	if output.Result.TruncatedAt != nil || !output.Result.HasScores() {
		return fmt.Errorf("token windows returned partial or unscored spans")
	}
	if len(output.Windows) > 0 {
		if err := tasks.ValidateWindowCoverage(output.ContentTokens, output.Windows, output.Result.Input); err != nil {
			return err
		}
	}
	return validateSpans(input.Text, output.Result)
}

func validateGroundedSpans(input tasks.GroundedTextRequest, output tasks.TokenClassificationResult) error {
	return validateSpans(input.Answer, output)
}

// validateSpans checks that every span is an ordered, non-overlapping UTF-8
// byte range of the text it was read from, with its text and probability.
func validateSpans(text string, result tasks.TokenClassificationResult) error {
	if !utf8.ValidString(text) {
		return fmt.Errorf("token span input must be valid UTF-8")
	}
	limit := len(text)
	if result.TruncatedAt != nil {
		limit = *result.TruncatedAt
		if limit < 0 || limit > len(text) || !utf8.ValidString(text[:limit]) {
			return fmt.Errorf("token span truncation must be an input byte boundary")
		}
	}
	previousEnd := 0
	for _, span := range result.Entities {
		if span.EntityType == "" || span.Start < previousEnd || span.End <= span.Start || span.End > limit || text[span.Start:span.End] != span.Text {
			return fmt.Errorf("token span does not match its input byte range")
		}
		previousEnd = span.End
		if result.HasScores() && (math.IsNaN(float64(span.Confidence)) || span.Confidence < 0 || span.Confidence > 1) {
			return fmt.Errorf("token span probability is outside [0,1]")
		}
	}
	if result.Summary != nil {
		if result.SummarySemantics == nil || result.SummarySemantics.Unit == "" {
			return fmt.Errorf("token aggregate score semantics are unavailable")
		}
		if err := result.SummarySemantics.Validate(*result.Summary); err != nil {
			return err
		}
	}
	return nil
}
