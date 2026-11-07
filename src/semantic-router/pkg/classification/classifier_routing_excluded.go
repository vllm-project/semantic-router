package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

// Jailbreak, PII and content-safety signals scan the text routing_input left
// out next to the routing text, so a client cannot hide content from them
// inside a skipped message or a stripped block.

func safetyPieces(text string, routing routinginput.Outcome) []string {
	if len(routing.ExcludedCurrent) == 0 {
		return []string{text}
	}
	return append([]string{text}, routing.ExcludedCurrent...)
}

// safetyText joins the pieces for signals that read one text.
func safetyText(text string, routing routinginput.Outcome) string {
	if len(routing.ExcludedCurrent) == 0 {
		return text
	}
	pieces := make([]string, 0, len(routing.ExcludedCurrent)+1)
	for _, piece := range safetyPieces(text, routing) {
		if piece != "" {
			pieces = append(pieces, piece)
		}
	}
	return strings.Join(pieces, "\n\n")
}

// safetyPriorUserMessages returns the prior user messages as the client sent
// them when routing_input changed the request.
func safetyPriorUserMessages(prior []string, routing routinginput.Outcome) []string {
	if !routing.Changed() {
		return prior
	}
	return routing.ExcludedPrior
}
