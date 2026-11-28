package llmprotocol

import "slices"

// InstructionsAfterToolResults moves each system or developer message that
// sits between a tool call and its results to just after those results, and
// reports how many it moved. Responses accepts an instruction item there, but
// Chat and Messages backends need a call's results right after the call. A
// user or non-calling assistant message ends the span even when a call has
// no result, and the held instructions go right before it.
func InstructionsAfterToolResults(messages []Message) ([]Message, int) {
	if !slices.ContainsFunc(messages, isInstructionMessage) {
		return messages, 0
	}
	placed := make([]Message, 0, len(messages))
	type heldInstruction struct {
		message Message
		at      int
	}
	var held []heldInstruction
	moved := 0
	pending := map[string]bool{}
	release := func() {
		at := len(placed)
		for _, instruction := range held {
			if at > instruction.at {
				moved++
			}
			placed = append(placed, instruction.message)
		}
		held = nil
	}
	for _, message := range messages {
		switch {
		case isInstructionMessage(message):
			if len(pending) > 0 {
				held = append(held, heldInstruction{message: message, at: len(placed)})
				continue
			}
		case message.WithToolResults || slices.ContainsFunc(message.Content, isToolResult):
			placed = append(placed, message)
			for _, content := range message.Content {
				if isToolResult(content) {
					delete(pending, content.ToolResult.CallID)
				}
			}
			if len(pending) == 0 {
				release()
			}
			continue
		case message.Role == RoleAssistant && slices.ContainsFunc(message.Content, isToolCall):
			for _, content := range message.Content {
				if isToolCall(content) {
					pending[content.ToolCall.ID] = true
				}
			}
		default:
			clear(pending)
			release()
		}
		placed = append(placed, message)
	}
	release()
	return placed, moved
}

func isToolResult(content Content) bool {
	return content.Kind == ContentToolResult && content.ToolResult != nil
}

func isToolCall(content Content) bool {
	return content.Kind == ContentToolCall && content.ToolCall != nil
}

// ProjectInstructionsAfterToolResults preserves tool adjacency before a target
// encodes distinct turns, or an instruction becomes user text. The caller must
// report the placement diagnostic even if a later projection changes the role.
func ProjectInstructionsAfterToolResults(request Request, target WireFormat) (Request, Diagnostics) {
	messages, moved := InstructionsAfterToolResults(request.Messages)
	if moved == 0 {
		return request, nil
	}
	projected := request
	projected.Messages = messages
	projected.Generation++ // Native replay must follow the same placement as re-encoding.
	field := "messages"
	if request.Trusted.SourceFormat == OpenAIResponsesV1 {
		field = "input"
	}
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat, Target: target, Field: field,
		Action: DiagnosticApproximated,
		Reason: "a system or developer item between a tool call and its output moves after the output",
	}}
}
