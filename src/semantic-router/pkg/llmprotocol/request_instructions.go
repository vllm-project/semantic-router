package llmprotocol

import "slices"

// SetSystemInstruction applies a deterministic system instruction policy to the
// neutral request. insert prepends to the first system block; other modes
// replace every system and developer block the caller sent with one system
// block, matching the router's existing system_prompt contract. That includes
// the system messages an Anthropic Messages client puts inside the
// conversation.
func SetSystemInstruction(request *Request, text, mode string) bool {
	if request == nil || text == "" {
		return false
	}
	content := Content{Kind: ContentText, Text: text}
	if mode != "insert" {
		request.Instructions = []InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}
		request.Messages = withoutCallerInstructions(request.Messages)
		return true
	}
	for i := range request.Instructions {
		instruction := &request.Instructions[i]
		if instruction.Role != RoleSystem {
			continue
		}
		instruction.Content = append([]Content{content}, instruction.Content...)
		return true
	}
	request.Instructions = append([]InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}, request.Instructions...)
	return true
}

// withoutCallerInstructions drops the system and developer messages inside
// the conversation. One that also sets per-message effort keeps that control
// as an empty system turn, as ProjectAnthropicSystemMessages does. It builds a
// new slice, since the request may share its messages with another view.
func withoutCallerInstructions(messages []Message) []Message {
	if !slices.ContainsFunc(messages, isInstructionMessage) {
		return messages
	}
	kept := make([]Message, 0, len(messages))
	for _, message := range messages {
		if !isInstructionMessage(message) {
			kept = append(kept, message)
		} else if message.ReasoningEffort != "" {
			kept = append(kept, Message{ID: message.ID, Role: RoleSystem, ReasoningEffort: message.ReasoningEffort})
		}
	}
	return kept
}

func isInstructionMessage(message Message) bool {
	return message.Role == RoleSystem || message.Role == RoleDeveloper
}
