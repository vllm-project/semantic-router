package llmprotocol

// StartsConversationTurn reports whether a message opens a new conversation
// turn: a user message carrying at least one block that is not a tool result.
// Anthropic represents tool results as user messages; those continue the
// current turn. Every component that segments history into turns uses this
// rule, so evidence about a turn and edits to that turn agree on its bounds.
func StartsConversationTurn(message Message) bool {
	if message.Role != RoleUser {
		return false
	}
	for _, content := range message.Content {
		if content.Kind != ContentToolResult {
			return true
		}
	}
	return false
}
