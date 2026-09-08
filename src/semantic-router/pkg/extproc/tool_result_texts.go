package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// extractToolResultTexts returns textual blocks from tool results in request
// order. It deliberately preserves duplicates; request-scoped deduplication
// belongs to the PII collector, where content from all configured sources can
// share the same detector cache.
func extractToolResultTexts(req *llmprotocol.Request) []string {
	if req == nil {
		return nil
	}

	var texts []string
	for _, message := range req.Messages {
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
				continue
			}

			for _, resultContent := range content.ToolResult.Content {
				if resultContent.Kind != llmprotocol.ContentText || resultContent.Text == "" {
					continue
				}
				texts = append(texts, resultContent.Text)
			}
		}
	}

	return texts
}
