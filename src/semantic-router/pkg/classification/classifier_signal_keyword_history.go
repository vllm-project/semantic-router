package classification

import "strings"

// keywordHistoryMessageLimit bounds how many prior user messages are joined
// into the history text, so scan cost stays flat on long conversations.
const keywordHistoryMessageLimit = 20

// keywordHistoryText joins the most recent prior user messages and the
// current message, oldest first. Messages are separated by newlines so a
// keyword phrase cannot match across two messages; blank messages are skipped.
func keywordHistoryText(currentText string, priorUserMessages []string) string {
	if len(priorUserMessages) > keywordHistoryMessageLimit {
		priorUserMessages = priorUserMessages[len(priorUserMessages)-keywordHistoryMessageLimit:]
	}
	parts := make([]string, 0, len(priorUserMessages)+1)
	for _, msg := range priorUserMessages {
		if trimmed := strings.TrimSpace(msg); trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	if trimmed := strings.TrimSpace(currentText); trimmed != "" {
		parts = append(parts, trimmed)
	}
	return strings.Join(parts, "\n")
}
