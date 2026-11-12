package runtimetest

import (
	"maps"
	"slices"
	"strings"
)

// answerText is the text a model's Answer reads of a state: the state's text,
// or its parts in name order.
func answerText(state interface{}) string {
	parts := stateParts(state)
	texts := make([]string, 0, len(parts))
	for _, name := range slices.Sorted(maps.Keys(parts)) {
		texts = append(texts, parts[name])
	}
	return strings.Join(texts, "\n\n")
}
