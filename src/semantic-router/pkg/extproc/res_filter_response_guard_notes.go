package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

// hallucinationWarningText is the note on an answer a hallucination rule
// matched: the detector's score and the unsupported spans, with a severity
// and an explanation each when the detector explains its spans.
func hallucinationWarningText(observation *classification.ResponseHallucinationObservation) string {
	if observation == nil || observation.Evidence == nil {
		return "[Hallucination Warning] This response may contain unsupported claims. Please verify the information independently."
	}
	evidence := observation.Evidence
	warning := hallucinationWarningPrefix(evidence.Score.Available, evidence.Score.Kind, evidence.Confidence)
	switch {
	case len(evidence.Explained) > 0:
		warning += " Detailed analysis:"
		for i, span := range evidence.Explained {
			warning += fmt.Sprintf(" [%d] \"%s\" (severity: %s)", i+1, span.Text, severityToString(span.Severity))
			if span.Explanation != "" {
				warning += " - " + span.Explanation
			}
		}
	case len(evidence.Spans) > 0:
		warning += fmt.Sprintf(" Unsupported spans: \"%s\".", strings.Join(evidence.Spans, "\", \""))
	}
	return warning + " Please verify the information independently."
}

func hallucinationWarningPrefix(available bool, kind string, score float32) string {
	prefix := "[Hallucination Warning] This response may contain unsupported claims"
	if !available {
		return prefix + "."
	}
	if kind == "probability" {
		return fmt.Sprintf("%s (confidence: %.0f%%).", prefix, score*100)
	}
	return fmt.Sprintf("%s (score: %.3f, kind: %s).", prefix, score, kind)
}

// severityToString converts severity level (0-4) to human-readable string
func severityToString(severity int) string {
	switch severity {
	case 0:
		return "low"
	case 1:
		return "low-medium"
	case 2:
		return "medium"
	case 3:
		return "high"
	case 4:
		return "critical"
	default:
		return "unknown"
	}
}
