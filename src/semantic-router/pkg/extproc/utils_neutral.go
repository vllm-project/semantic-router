package extproc

import (
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// semanticAssistantContent returns the ordered, client-visible assistant text
// from the primary neutral output. Reasoning stays separate: guardrails and
// memory must not mistake hidden chain-of-thought for the delivered answer.
func semanticAssistantContent(response *llmprotocol.Response) string {
	if response == nil {
		return ""
	}
	var text strings.Builder
	for _, item := range response.Output {
		if item.Role != llmprotocol.RoleAssistant {
			continue
		}
		for _, content := range item.Content {
			switch content.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal:
				text.WriteString(content.Text)
			}
		}
	}
	return text.String()
}

// prependSemanticAssistantText mutates every assistant alternative in place so
// a response warning remains protocol-neutral until the inbound codec renders
// it. It returns false when the response has no client-visible text block.
func prependSemanticAssistantText(response *llmprotocol.Response, prefix string) bool {
	if response == nil || prefix == "" {
		return false
	}
	changed := prependSemanticOutputText(response.Output, prefix)
	for index := range response.Alternatives {
		changed = prependSemanticOutputText(response.Alternatives[index], prefix) || changed
	}
	if changed {
		response.Generation++
	}
	return changed
}

func prependSemanticOutputText(output []llmprotocol.OutputItem, prefix string) bool {
	changed := false
	for itemIndex := range output {
		if output[itemIndex].Role != llmprotocol.RoleAssistant {
			continue
		}
		for contentIndex := range output[itemIndex].Content {
			content := &output[itemIndex].Content[contentIndex]
			if content.Kind != llmprotocol.ContentText && content.Kind != llmprotocol.ContentRefusal {
				continue
			}
			content.Text = prefix + "\n\n" + content.Text
			changed = true
		}
	}
	return changed
}

// extractSemanticRequestSignals derives routing facts from the neutral request
// accepted at ingress. Wire-specific JSON walkers are confined to codecs.
func extractSemanticRequestSignals(request *llmprotocol.Request) *requestSignalSnapshot {
	result := &requestSignalSnapshot{Model: request.Model, Stream: request.Stream}
	if len(request.Metadata) > 0 {
		result.Metadata = make(map[string]string, len(request.Metadata))
		for key, value := range request.Metadata {
			result.Metadata[key] = value
		}
	}
	for _, instruction := range request.Instructions {
		text := semanticText(instruction.Content)
		if instruction.Role == llmprotocol.RoleDeveloper {
			result.HasDeveloperMessage = true
		} else {
			result.SystemMessageCount++
		}
		recordNonUserSignalMessage(result, string(instruction.Role), text)
		consumeNeutralContext(result, instruction.Content)
	}
	for _, message := range request.Messages {
		consumeSemanticMessage(result, message)
	}
	result.ToolDefinitionCount = len(request.Tools)
	for _, tool := range request.Tools {
		addContextBytes(result, len(tool.Name)+len(tool.Description), len(tool.InputSchema))
	}
	result.ContextTokenFloor = neutralContextTokenFloor(result, len(request.Messages)+len(request.Instructions))
	result.ContextEquivalentBytes = saturatingNeutralMultiply(result.ContextTokenFloor, classification.RequestContextBytesPerToken)
	return result
}

//nolint:gocognit,cyclop,funlen // Signal extraction walks the complete neutral message content vocabulary.
func consumeSemanticMessage(result *requestSignalSnapshot, message llmprotocol.Message) {
	previousWasTool := result.LastMessageToolResult || result.LastMessageRole == string(llmprotocol.RoleTool)
	role := string(message.Role)
	text := semanticText(message.Content)
	result.LastMessageRole = role
	result.LastMessageToolResult = false
	result.LastMessageFlowToolResult = false
	result.LastAssistantToolCall = false
	result.LastUserAfterToolResult = false
	switch message.Role {
	case llmprotocol.RoleUser:
		result.UserMessageCount++
		result.LastUserAfterToolResult = previousWasTool
		if text != "" {
			if result.UserContent != "" {
				result.PriorUserMessages = append(result.PriorUserMessages, result.UserContent)
			}
			result.UserContent = text
		}
	case llmprotocol.RoleAssistant:
		result.AssistantMessageCount++
		result.HasAssistantReply = result.HasAssistantReply || text != ""
		recordNonUserSignalMessage(result, role, text)
	case llmprotocol.RoleSystem:
		result.SystemMessageCount++
		recordNonUserSignalMessage(result, role, text)
	case llmprotocol.RoleDeveloper:
		result.HasDeveloperMessage = true
		recordNonUserSignalMessage(result, role, text)
	case llmprotocol.RoleTool:
		result.ToolMessageCount++
		result.LastMessageToolResult = true
	}
	for _, content := range message.Content {
		switch content.Kind {
		case llmprotocol.ContentImage:
			result.ImageContentCount++
			if result.FirstImageURL == "" {
				result.FirstImageURL = content.URL
			}
		case llmprotocol.ContentToolCall:
			result.AssistantToolCallCount++
			if message.Role == llmprotocol.RoleAssistant {
				result.LastAssistantToolCall = true
			}
			if content.ToolCall != nil {
				if content.ToolCall.Name != "" {
					result.AssistantToolNames = append(result.AssistantToolNames, content.ToolCall.Name)
				}
				addContextBytes(result, 0, len(content.ToolCall.Name)+len(content.ToolCall.Arguments))
			}
		case llmprotocol.ContentToolResult:
			result.ToolResultCount++
			result.LastMessageToolResult = true
			if content.ToolResult != nil && looper.IsWorkflowToolCallID(content.ToolResult.CallID) {
				result.LastMessageFlowToolResult = true
			}
		}
	}
	consumeNeutralContext(result, message.Content)
}

func consumeNeutralContext(result *requestSignalSnapshot, contents []llmprotocol.Content) {
	for _, content := range contents {
		switch content.Kind {
		case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
			addContextBytes(result, len(content.Text), 0)
		case llmprotocol.ContentImage:
			result.ContextHasNonText = true
		case llmprotocol.ContentAudio, llmprotocol.ContentVideo, llmprotocol.ContentFile:
			result.ContextHasNonText = true
			addContextBytes(result, 0, len(content.URL)+len(content.Data)+len(content.FileID))
		case llmprotocol.ContentToolResult:
			result.ContextHasNonText = true
			if content.ToolResult != nil {
				consumeNeutralContext(result, content.ToolResult.Content)
			}
		}
	}
}

func addContextBytes(result *requestSignalSnapshot, text, structured int) {
	result.ContextTextBytes = saturatingNeutralAdd(result.ContextTextBytes, text)
	result.ContextEquivalentBytes = saturatingNeutralAdd(result.ContextEquivalentBytes, structured)
}

func neutralContextTokenFloor(result *requestSignalSnapshot, messages int) int {
	textTokens := (result.ContextTextBytes + classification.RequestContextBytesPerToken - 1) / classification.RequestContextBytesPerToken
	structured := result.ContextEquivalentBytes
	images := saturatingNeutralMultiply(result.ImageContentCount, classification.RequestContextImageTokenBudget)
	framing := saturatingNeutralMultiply(messages, classification.RequestContextMessageFramingTokens)
	framing = saturatingNeutralAdd(framing, saturatingNeutralMultiply(result.AssistantToolCallCount, classification.RequestContextToolCallFramingTokens))
	framing = saturatingNeutralAdd(framing, saturatingNeutralMultiply(result.ToolDefinitionCount, classification.RequestContextToolDefinitionFramingTokens))
	return saturatingNeutralAdd(saturatingNeutralAdd(textTokens, structured), saturatingNeutralAdd(images, framing))
}

func saturatingNeutralAdd(left, right int) int {
	if right > 0 && left > math.MaxInt-right {
		return math.MaxInt
	}
	return left + right
}

func saturatingNeutralMultiply(left, right int) int {
	if left <= 0 || right <= 0 {
		return 0
	}
	if left > math.MaxInt/right {
		return math.MaxInt
	}
	return left * right
}
