package classification

// Includes prior user turns (not just system/assistant) so include_history
// detectors catch secrets from earlier turns. Security fix, issue #1961.
func historyForHistoryAwareSignals(priorUserMessages, nonUserMessages []string) []string {
	merged := make([]string, 0, len(priorUserMessages)+len(nonUserMessages))
	seen := make(map[string]struct{}, len(priorUserMessages)+len(nonUserMessages))
	for _, group := range [][]string{nonUserMessages, priorUserMessages} {
		for _, msg := range group {
			if msg == "" {
				continue
			}
			if _, ok := seen[msg]; ok {
				continue
			}
			seen[msg] = struct{}{}
			merged = append(merged, msg)
		}
	}
	return merged
}
