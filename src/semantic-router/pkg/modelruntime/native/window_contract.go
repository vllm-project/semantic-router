package native

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func validateWindowInput(input tasks.TextWindowsRequest) error {
	if err := validateText(input.Text); err != nil {
		return err
	}
	return tasks.ValidateTextWindows(input)
}

func validateWindowDistribution(_ tasks.TextWindowsRequest, result tasks.WindowedLabelDistribution) error {
	spans := make([][2]int, len(result.Windows))
	classes := 0
	for i, window := range result.Windows {
		if i == 0 {
			classes = len(window.Probabilities)
		}
		if len(window.Probabilities) != classes {
			return fmt.Errorf("window class counts differ")
		}
		if err := validateDistribution("", tasks.LabelDistribution{Probabilities: window.Probabilities}); err != nil {
			return err
		}
		spans[i] = [2]int{window.Start, window.End}
	}
	return tasks.ValidateWindowCoverage(result.ContentTokens, spans, result.Input)
}

func validateWindowScores(_ tasks.TextWindowsRequest, result tasks.WindowedLabelScores) error {
	spans := make([][2]int, len(result.Windows))
	classes := 0
	for i, window := range result.Windows {
		if i == 0 {
			classes = len(window.Scores)
		}
		if len(window.Scores) != classes {
			return fmt.Errorf("window label counts differ")
		}
		if err := tasks.ValidateLabelScores(window.Scores); err != nil {
			return err
		}
		spans[i] = [2]int{window.Start, window.End}
	}
	return tasks.ValidateWindowCoverage(result.ContentTokens, spans, result.Input)
}
