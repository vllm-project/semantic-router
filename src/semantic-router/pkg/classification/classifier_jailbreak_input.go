package classification

import "strings"

// JailbreakContent preserves the request role of untrusted content.
// Roles describe the input boundary; they are not identity or authorization.
type JailbreakContent struct {
	Role       string
	Text       string
	Incomplete bool // Some of this content could not be read as text.
}

// JailbreakInput is a dedicated, uncompressed projection of the neutral request.
// A non-nil empty projection means there is no eligible content, rather than an
// invitation to fall back to system, developer, or assistant instructions.
type JailbreakInput struct {
	Current []JailbreakContent
	History []JailbreakContent
}

// withJailbreakInput keeps coverage scoped to the same roles and turns as the
// projected text. The general tool-result facts can include earlier turns.
func (r systemOneReading) withJailbreakInput(input *JailbreakInput) systemOneReading {
	current, incomplete := jailbreakRoleText(input.Current, "user")
	r.request, r.requestIncomplete = strings.Join(current, "\n\n"), incomplete
	r.history, r.historyIncomplete = jailbreakRoleText(input.History, "user")
	r.toolResults, r.toolResultsIncomplete = jailbreakRoleText(input.Current, "tool")
	return r
}

func jailbreakRoleText(contents []JailbreakContent, role string) ([]string, bool) {
	texts := make([]string, 0, len(contents))
	incomplete := false
	for _, content := range contents {
		if content.Role != role {
			continue
		}
		incomplete = incomplete || content.Incomplete
		if content.Text != "" {
			texts = append(texts, content.Text)
		}
	}
	return texts, incomplete
}
