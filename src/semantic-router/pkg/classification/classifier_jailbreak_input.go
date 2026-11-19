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

// jailbreakInputTexts is the text of the contents in one role: "user" for the
// turns a jailbreak rule reads, "tool" for the tool results it reads when its
// input asks for them.
func jailbreakInputTexts(contents []JailbreakContent, role string) []string {
	texts := make([]string, 0, len(contents))
	for _, content := range contents {
		if content.Text != "" && content.Role == role {
			texts = append(texts, content.Text)
		}
	}
	return texts
}
