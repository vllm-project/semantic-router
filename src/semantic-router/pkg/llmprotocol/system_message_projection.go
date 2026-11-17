package llmprotocol

import "fmt"

// ProjectAnthropicSystemMessages sends the Anthropic system messages inside the
// conversation to Chat and Responses backends as user text in the same place.
// Many open-model chat templates reject a system message after the first
// position, and others silently drop it. Claude Code inserts these messages at
// the point they refer to (a file changed on disk, a truncated read, a new
// working directory), so the text joins the user message right before it, or
// becomes a user message of its own after a tool result or an assistant turn.
// Earlier turns are never rewritten, which keeps the conversation's cached
// prefix. The caller applies it only when the operator lists
// LossyFieldSystemMessageRole and must report the returned diagnostic. A
// message that also sets per-message effort keeps an empty system turn, so
// that control is still reported as dropped by the encoder.
func ProjectAnthropicSystemMessages(request Request, target WireFormat) (Request, Diagnostics) {
	if request.Trusted.SourceFormat != AnthropicMessagesV1 ||
		(target != OpenAIChatV1 && target != OpenAIResponsesV1) {
		return request, nil
	}

	messages := make([]Message, 0, len(request.Messages))
	count := 0
	for _, message := range request.Messages {
		if message.Role != RoleSystem || len(message.Content) == 0 {
			messages = append(messages, message)
			continue
		}
		count++
		if last := len(messages) - 1; last >= 0 && messages[last].Role == RoleUser {
			messages[last].Content = append(append([]Content(nil), messages[last].Content...), message.Content...)
		} else {
			messages = append(messages, Message{Role: RoleUser, Content: message.Content})
		}
		if message.ReasoningEffort != "" {
			messages = append(messages, Message{ID: message.ID, Role: RoleSystem, ReasoningEffort: message.ReasoningEffort})
		}
	}
	if count == 0 {
		return request, nil
	}

	projected := request
	projected.Messages = messages
	projected.Generation++ // The source envelope still places these messages in the conversation.
	return projected, Diagnostics{{
		Source: AnthropicMessagesV1,
		Target: target,
		Field:  LossyFieldSystemMessageRole,
		Action: DiagnosticApproximated,
		Reason: fmt.Sprintf("%d system messages sent as user text in place", count),
	}}
}
