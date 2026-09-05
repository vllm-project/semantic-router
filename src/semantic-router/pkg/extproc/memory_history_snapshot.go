package extproc

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cloneSemanticMessages detaches all mutable history fields before they are
// handed to asynchronous persistence. Strings can safely share backing data.
func cloneSemanticMessages(messages []llmprotocol.Message) []llmprotocol.Message {
	result := slices.Clone(messages)
	for index := range result {
		result[index].Content = cloneMemoryContents(messages[index].Content)
	}
	return result
}

func cloneMemoryContents(contents []llmprotocol.Content) []llmprotocol.Content {
	result := slices.Clone(contents)
	for index := range result {
		content := &result[index]
		content.Citations = slices.Clone(content.Citations)
		content.Cache = cloneMemoryValue(content.Cache)
		content.ToolCall = cloneMemoryValue(content.ToolCall)
		if content.ToolResult != nil {
			content.ToolResult = cloneMemoryValue(content.ToolResult)
			content.ToolResult.Content = cloneMemoryContents(content.ToolResult.Content)
			content.ToolResult.IsError = cloneMemoryValue(content.ToolResult.IsError)
		}
		if content.GeneratedImage != nil {
			content.GeneratedImage = cloneMemoryValue(content.GeneratedImage)
			content.GeneratedImage.Result = cloneMemoryValue(content.GeneratedImage.Result)
			content.GeneratedImage.PartialIndex = cloneMemoryValue(content.GeneratedImage.PartialIndex)
		}
	}
	return result
}

// cloneMemoryValue copies a pointee; callers detach any nested mutable fields.
func cloneMemoryValue[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
