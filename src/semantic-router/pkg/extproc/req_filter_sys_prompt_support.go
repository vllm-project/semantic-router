package extproc

func extractExistingSystemMessage(messages []interface{}) (string, bool) {
	if len(messages) == 0 {
		return "", false
	}

	firstMessage, ok := messages[0].(map[string]interface{})
	if !ok {
		return "", false
	}

	role, ok := firstMessage["role"].(string)
	if !ok || role != "system" {
		return "", false
	}

	content, _ := firstMessage["content"].(string)
	return content, true
}

func resolveSystemPromptInjection(systemPrompt, mode, existingSystemContent string, hasSystemMessage bool) (string, string) {
	switch mode {
	case "insert":
		if hasSystemMessage {
			return systemPrompt + "\n\n" + existingSystemContent, "Inserted category-specific system prompt before existing system message"
		}
		return systemPrompt, "Added category-specific system prompt (insert mode, no existing system message)"
	default:
		if hasSystemMessage {
			return systemPrompt, "Replaced existing system message with category-specific system prompt"
		}
		return systemPrompt, "Added category-specific system prompt to the beginning of messages"
	}
}

func upsertLeadingSystemMessage(messages []interface{}, finalSystemContent string, hasSystemMessage bool) []interface{} {
	systemMessage := map[string]interface{}{
		"role":    "system",
		"content": finalSystemContent,
	}
	if hasSystemMessage {
		messages[0] = systemMessage
		return messages
	}
	return append([]interface{}{systemMessage}, messages...)
}
