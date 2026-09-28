package classification

// JailbreakContent preserves the request role of one untrusted text piece.
// Roles describe the input boundary; they are not identity or authorization.
type JailbreakContent struct {
	Role string
	Text string
}

// JailbreakInput is a dedicated, uncompressed projection of the neutral request.
// A non-nil empty projection means there is no eligible content, rather than an
// invitation to fall back to system, developer, or assistant instructions.
type JailbreakInput struct {
	Current []JailbreakContent
	History []JailbreakContent
}

func jailbreakInputTexts(contents []JailbreakContent) []string {
	texts := make([]string, 0, len(contents))
	for _, content := range contents {
		if content.Text != "" && (content.Role == "user" || content.Role == "tool") {
			texts = append(texts, content.Text)
		}
	}
	return texts
}
