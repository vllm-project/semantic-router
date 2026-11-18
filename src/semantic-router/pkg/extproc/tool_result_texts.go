package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// toolResultTextExtraction describes the portion of tool-result content that
// the text-only readers (the pii, systemone and keyword rules with
// input.tool_results) could inspect. Each text is one tool result, so a rule
// evaluates a tool result as a unit regardless of how many text blocks carry
// it. Texts deliberately preserve duplicates; request-scoped deduplication
// belongs to the PII collector, where content from all configured sources can
// share the same detector cache.
type toolResultTextExtraction struct {
	texts         []string
	incomplete    bool
	skippedBlocks int
}

// extractToolResultTexts returns the text of each tool result in request
// order, joining a tool result's text blocks with a newline, and reports
// whether any tool-result block was not inspectable by the text-only
// readers. A request with no tool results is complete; a tool result
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

			var blocks []string
			for _, resultContent := range content.ToolResult.Content {
				switch resultContent.Kind {
				case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
					if resultContent.Text != "" {
						blocks = append(blocks, resultContent.Text)
					}
				default:
					extraction.incomplete = true
					extraction.skippedBlocks++
				}
			}
			if len(blocks) > 0 {
				extraction.texts = append(extraction.texts, strings.Join(blocks, "\n"))
			}
		}
	}

	return extraction
}
