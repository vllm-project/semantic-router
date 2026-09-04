package tools

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// SemanticTool projects the typed tool-catalog value into the protocol-neutral
// runtime contract. Only the JSON Schema map crosses this boundary as JSON;
// the enclosing provider object is never round-tripped through a wire shape.
func SemanticTool(tool openai.ChatCompletionToolParam) (llmprotocol.Tool, error) {
	if _, overridden := tool.Overrides(); overridden || len(tool.ExtraFields()) > 0 {
		return llmprotocol.Tool{}, fmt.Errorf("tool catalog value contains an opaque provider override")
	}
	if _, overridden := tool.Function.Overrides(); overridden || len(tool.Function.ExtraFields()) > 0 {
		return llmprotocol.Tool{}, fmt.Errorf("tool function contains an opaque provider override")
	}
	toolType := string(tool.Type)
	if toolType != "" && toolType != "function" {
		return llmprotocol.Tool{}, fmt.Errorf("unsupported tool type %q", toolType)
	}

	inputSchema := json.RawMessage(`{"type":"object"}`)
	if len(tool.Function.Parameters) > 0 {
		encoded, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			return llmprotocol.Tool{}, fmt.Errorf("encode tool %q input schema: %w", tool.Function.Name, err)
		}
		inputSchema = encoded
	}

	var strict *bool
	if tool.Function.Strict.Valid() {
		value := tool.Function.Strict.Value
		strict = &value
	}
	return llmprotocol.Tool{
		Name:        tool.Function.Name,
		Description: tool.Function.Description.Or(""),
		Strict:      strict,
		InputSchema: append(json.RawMessage(nil), inputSchema...),
	}, nil
}

func SemanticTools(values []openai.ChatCompletionToolParam) ([]llmprotocol.Tool, error) {
	result := make([]llmprotocol.Tool, 0, len(values))
	for _, value := range values {
		tool, err := SemanticTool(value)
		if err != nil {
			return nil, err
		}
		result = append(result, tool)
	}
	return result, nil
}

func SemanticToolsFromCandidates(values []ToolSimilarity, limit int) ([]llmprotocol.Tool, error) {
	if limit <= 0 || limit > len(values) {
		limit = len(values)
	}
	result := make([]llmprotocol.Tool, 0, limit)
	for index := 0; index < limit; index++ {
		tool, err := SemanticTool(values[index].Entry.Tool)
		if err != nil {
			return nil, err
		}
		result = append(result, tool)
	}
	return result, nil
}
