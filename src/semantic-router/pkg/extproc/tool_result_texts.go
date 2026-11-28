package extproc

import (
	"encoding/json"
	"maps"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
			if content.Kind == llmprotocol.ContentServerTool {
				extraction.addServerToolResult(content.Raw)
				continue
			}
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

// serverToolResultMetadata are the fields of an Anthropic server tool block
// that name or describe it rather than carry what the tool returned.
var serverToolResultMetadata = map[string]bool{
	"type": true, "id": true, "tool_use_id": true, "url": true, "title": true,
	"page_age": true, "media_type": true, "retrieved_at": true, "file_id": true,
}

// addServerToolResult reads the text an Anthropic server tool returned, such
// as a fetched page or a code run's output, which the codec keeps as written.
// Its call, server_tool_use, carries no result. A base64 document, such as a
// fetched PDF, and a search result's encrypted_content reach the model but
// not the detector, so they leave the scan incomplete.
func (e *toolResultTextExtraction) addServerToolResult(raw json.RawMessage) {
	var block map[string]any
	if json.Unmarshal(raw, &block) != nil {
		e.incomplete = true
		e.skippedBlocks++
		return
	}
	if block["type"] == "server_tool_use" {
		return
	}
	unread := false
	var walk func(value any)
	walk = func(value any) {
		switch value := value.(type) {
		case string:
			if value != "" {
				e.texts = append(e.texts, value)
			}
		case []any:
			for _, item := range value {
				walk(item)
			}
		case map[string]any:
			if _, encrypted := value["encrypted_content"]; encrypted || value["type"] == "base64" {
				unread = true
				return
			}
			for _, key := range slices.Sorted(maps.Keys(value)) {
				if !serverToolResultMetadata[key] {
					walk(value[key])
				}
			}
		}
	}
	walk(block)
	if unread {
		e.incomplete = true
		e.skippedBlocks++
	}
}
