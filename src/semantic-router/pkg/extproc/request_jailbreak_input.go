package extproc

import (
	"slices"

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
	direct := classification.JailbreakContent{Role: string(message.Role), Text: semanticText(message.Content)}
	var toolResults []classification.JailbreakContent
	for _, content := range message.Content {
		switch content.Kind {
		case llmprotocol.ContentText:
		case llmprotocol.ContentToolResult:
			result := jailbreakToolResult(content.ToolResult)
			if result.Text != "" || result.Incomplete {
				toolResults = append(toolResults, result)
			}
		default:
			direct.Incomplete = true
		}
	}
	if direct.Text == "" && !direct.Incomplete {
		return toolResults
	}
	return append([]classification.JailbreakContent{direct}, toolResults...)
}

func jailbreakToolResult(result *llmprotocol.ToolResult) classification.JailbreakContent {
	content := classification.JailbreakContent{Role: string(llmprotocol.RoleTool)}
	if result == nil {
		content.Incomplete = true
		return content
	}
	content.Text = semanticText(result.Content)
	content.Incomplete = slices.ContainsFunc(result.Content, func(block llmprotocol.Content) bool {
		return block.Kind != llmprotocol.ContentText
	})
	return content
}
