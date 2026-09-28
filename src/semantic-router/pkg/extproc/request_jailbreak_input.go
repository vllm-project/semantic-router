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
	// A codec may split one wire message into user text and several tool-result
	// messages. Keep the trailing untrusted group together so a sibling tool
	// result cannot disappear merely because include_history is disabled.
	currentStart := len(request.Messages)
	for currentStart > 0 {
		role := request.Messages[currentStart-1].Role
		if role != llmprotocol.RoleUser && role != llmprotocol.RoleTool {
			break
		}
		currentStart--
	}
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
