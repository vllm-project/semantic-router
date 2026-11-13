package protocolcodec

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenAI request formats have no equivalent of Anthropic's tool-result failure
// flag. Check the neutral value (including mutations), not just the source format.
// Permissive translation keeps the result payload unchanged and reports the loss
// once per request; it must not invent error text or send an unsupported wire field.
func appendToolResultErrorLoss(diagnostics *llmprotocol.Diagnostics, request llmprotocol.Request, policy llmprotocol.Policy, target llmprotocol.WireFormat) error {
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil && content.ToolResult.IsError != nil && *content.ToolResult.IsError {
				return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, target,
					llmprotocol.LossyFieldToolResultError, "target format cannot represent tool-result failure status")
			}
		}
	}
	return nil
}

// toolResultMediaAsUser reports whether images and files in tool results may
// move to a user message, the only place Chat Completions accepts them. Without
// the opt-in the Chat encoder keeps rejecting them.
func toolResultMediaAsUser(policy llmprotocol.Policy) bool {
	return policy.LossyFeatures != llmprotocol.LossyReject ||
		slices.Contains(policy.AllowedLossyFields, llmprotocol.LossyFieldToolResultContent)
}

func appendToolResultMediaLoss(diagnostics *llmprotocol.Diagnostics, request llmprotocol.Request, policy llmprotocol.Policy) error {
	if !toolResultMediaAsUser(policy) {
		return nil
	}
	for _, message := range request.Messages {
		if _, media := splitToolResultMedia(message); len(media) > 0 {
			return appendLossy(diagnostics, policy, request.Trusted.SourceFormat, llmprotocol.OpenAIChatV1,
				llmprotocol.LossyFieldToolResultContent, "tool-result images and files sent as a user message after the results")
		}
	}
	return nil
}

// toolResultMovedText stands in for a tool result whose content all moved, so
// the tool message is not empty and keeps any cache boundary set on it.
const toolResultMovedText = "The tool result is in the next message."

// splitToolResultMedia keeps the text of a message's tool results in place and
// returns their other content, in order.
func splitToolResultMedia(message llmprotocol.Message) (llmprotocol.Message, []llmprotocol.Content) {
	var media []llmprotocol.Content
	var split []llmprotocol.Content
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
			split = append(split, content)
			continue
		}
		result := *content.ToolResult
		result.Content = nil
		for _, part := range content.ToolResult.Content {
			if part.Kind == llmprotocol.ContentText {
				result.Content = append(result.Content, part)
			} else {
				media = append(media, part)
			}
		}
		if len(result.Content) == 0 && len(content.ToolResult.Content) > 0 {
			result.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: toolResultMovedText}}
		}
		content.ToolResult = &result
		split = append(split, content)
	}
	if len(media) == 0 {
		return message, nil
	}
	message.Content = split
	return message, media
}
