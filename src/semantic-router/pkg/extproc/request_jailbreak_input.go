package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// extractJailbreakInput keeps attack evidence separate from trusted instruction
// and assistant context. It neither rewrites the provider request nor changes
// other signals' history. Tool results remain untrusted even when their wire
// protocol represents them inside a user message.
func extractJailbreakInput(request *llmprotocol.Request) *classification.JailbreakInput {
	input := &classification.JailbreakInput{}
	currentStart := currentTurnStart(request.Messages)
	for index, message := range request.Messages {
		contents := jailbreakMessageContents(message)
		if index >= currentStart {
			input.Current = append(input.Current, contents...)
		} else {
			input.History = append(input.History, contents...)
		}
	}
	return input
}

// currentTurnStart is where the request's current turn starts: the user and
// tool messages after the last assistant message. A codec may split one wire
// message into user text and several tool-result messages, so the trailing
// untrusted group stays together and a sibling tool result cannot disappear
// merely because include_history is disabled. A system or developer message
// inside the conversation, which Claude Code inserts where it applies, after
// the user's message or a tool result, does not end the turn.
func currentTurnStart(messages []llmprotocol.Message) int {
	start := len(messages)
	for index := len(messages) - 1; index >= 0; index-- {
		role := messages[index].Role
		if isInstructionRole(role) {
			continue
		}
		if role != llmprotocol.RoleUser && role != llmprotocol.RoleTool {
			break
		}
		start = index
	}
	return start
}

func isInstructionRole(role llmprotocol.Role) bool {
	return role == llmprotocol.RoleSystem || role == llmprotocol.RoleDeveloper
}

func jailbreakMessageContents(message llmprotocol.Message) []classification.JailbreakContent {
	if message.Role != llmprotocol.RoleUser && message.Role != llmprotocol.RoleTool {
		return nil
	}
	var contents []classification.JailbreakContent
	if text := semanticText(message.Content); text != "" {
		contents = append(contents, classification.JailbreakContent{Role: string(message.Role), Text: text})
	}
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
			continue
		}
		if text := semanticText(content.ToolResult.Content); text != "" {
			contents = append(contents, classification.JailbreakContent{Role: "tool", Text: text})
		}
	}
	return contents
}
