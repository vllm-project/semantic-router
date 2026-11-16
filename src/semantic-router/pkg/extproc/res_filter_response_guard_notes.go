package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// hallucinationFinding is what the hallucination rules a guard read made of
// an answer: whether one was asked, answered or matched, the highest
// probability of an unsupported claim among those that answered, and the
// texts of the spans the matched rules found.
type hallucinationFinding struct {
	asked       bool
	answered    bool
	detected    bool
	probability float64
	spans       []string
}

func findHallucination(observed *classification.ResponseSignal) hallucinationFinding {
	var finding hallucinationFinding
	if observed == nil {
		return finding
	}
	prefix := config.SignalTypeHallucination + ":"
	for key, value := range observed.Values {
		if rule, ok := strings.CutPrefix(key, prefix); ok && !strings.Contains(rule, ":") {
			finding.asked, finding.answered = true, true
			finding.probability = max(finding.probability, value)
		}
	}
	for key := range observed.Errors {
		if strings.HasPrefix(key, prefix) {
			finding.asked = true
		}
	}
	for _, reference := range observed.Matched[config.SignalTypeHallucination] {
		if strings.Contains(reference, ":") {
			continue
		}
		finding.detected = true
		for _, span := range observed.Spans[signalKey(config.SignalTypeHallucination, reference)] {
			finding.spans = append(finding.spans, span.Text)
		}
	}
	return finding
}

// hallucinationWarningText is the note on an answer a hallucination rule
// matched: the highest probability of an unsupported claim and the spans the
// matched rules found.
func hallucinationWarningText(observed *classification.ResponseSignal) string {
	finding := findHallucination(observed)
	if !finding.detected {
		return "[Hallucination Warning] This response may contain unsupported claims. Please verify the information independently."
	}
	warning := fmt.Sprintf("[Hallucination Warning] This response may contain unsupported claims (confidence: %.0f%%).", finding.probability*100)
	if len(finding.spans) > 0 {
		warning += fmt.Sprintf(" Unsupported spans: \"%s\".", strings.Join(finding.spans, "\", \""))
	}
	return warning + " Please verify the information independently."
}
