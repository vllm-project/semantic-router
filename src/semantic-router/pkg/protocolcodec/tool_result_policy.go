package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// OpenAI request formats have no equivalent of Anthropic's tool-result failure
// flag. Check the neutral value (including mutations), not just the source format.
// Permissive translation keeps the result payload unchanged and reports the loss
// once per request; it must not invent error text or send an unsupported wire field.
func appendToolResultErrorLoss(diagnostics *llmprotocol.Diagnostics, request llmprotocol.Request, policy llmprotocol.Policy, target llmprotocol.WireFormat) error {
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil && content.ToolResult.IsError != nil && *content.ToolResult.IsError {
				return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, target,
					"tool_result.is_error", "target format cannot represent tool-result failure status")
			}
		}
	}
	return nil
}
