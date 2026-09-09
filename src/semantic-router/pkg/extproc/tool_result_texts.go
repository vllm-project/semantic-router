package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// toolResultTextExtraction describes the portion of tool-result content that
// the text-only PII detector could inspect. Texts deliberately preserve
// duplicates; request-scoped deduplication belongs to the PII collector, where
// content from all configured sources can share the same detector cache.
type toolResultTextExtraction struct {
	texts         []string
	incomplete    bool
	skippedBlocks int
}

// extractToolResultTexts returns textual blocks from tool results in request
// order and reports whether any tool-result block was not inspectable by the
// text-only detector. A request with no tool results is complete; a tool result
// containing only unsupported content is incomplete even when no text exists.
func extractToolResultTexts(req *llmprotocol.Request) toolResultTextExtraction {
	if req == nil {
		return toolResultTextExtraction{}
	}

	var extraction toolResultTextExtraction
	for _, message := range req.Messages {
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentToolResult {
				continue
			}
			if content.ToolResult == nil {
				extraction.incomplete = true
				extraction.skippedBlocks++
				continue
			}

			for _, resultContent := range content.ToolResult.Content {
				switch resultContent.Kind {
				case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
					if resultContent.Text != "" {
						extraction.texts = append(extraction.texts, resultContent.Text)
					}
				default:
					extraction.incomplete = true
					extraction.skippedBlocks++
				}
			}
		}
	}

	return extraction
}
