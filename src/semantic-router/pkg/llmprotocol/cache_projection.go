package llmprotocol

import "fmt"

// ProjectAnthropicCacheDirectives omits the Anthropic prompt-cache boundaries
// a dispatch format cannot represent. Responses has no per-block cache
// directive, so every boundary goes. Chat Completions keeps the boundaries on
// tools, instructions and content parts, but tool calls, reasoning and refusals
// are message fields there and a tool result without content has no text part,
// so their boundaries go; Claude Code marks a tool_use block when it compacts.
// Cache controls are performance hints, not prompt content. The caller must
// surface the returned dropped diagnostic to the public client. Other
// source/target pairs retain the strict capability contract.
func ProjectAnthropicCacheDirectives(request Request, target WireFormat) (Request, Diagnostics) {
	if request.Trusted.SourceFormat != AnthropicMessagesV1 ||
		!RequiredCapabilities(request).Supports(CapabilityCacheDirectives) {
		return request, nil
	}
	switch target {
	case OpenAIResponsesV1:
		return projectResponsesCacheDirectives(request)
	case OpenAIChatV1:
		return projectChatCacheDirectives(request)
	default:
		return request, nil
	}
}

func projectResponsesCacheDirectives(request Request) (Request, Diagnostics) {
	projected := request
	dropped := 0
	projected.Tools = append([]Tool(nil), request.Tools...)
	for index := range projected.Tools {
		if projected.Tools[index].Cache != nil {
			projected.Tools[index].Cache = nil
			dropped++
		}
	}
	projected.Instructions = append([]InstructionBlock(nil), request.Instructions...)
	for index := range projected.Instructions {
		projected.Instructions[index].Content = contentWithoutCacheDirectives(request.Instructions[index].Content, &dropped)
	}
	projected.Messages = append([]Message(nil), request.Messages...)
	for index := range projected.Messages {
		projected.Messages[index].Content = contentWithoutCacheDirectives(request.Messages[index].Content, &dropped)
	}
	projected.Generation++ // The source envelope still contains cache_control.
	return projected, cacheDroppedDiagnostics(OpenAIResponsesV1,
		fmt.Sprintf("Responses cannot represent %d Anthropic prompt-cache boundaries", dropped))
}

func contentWithoutCacheDirectives(contents []Content, dropped *int) []Content {
	projected := append([]Content(nil), contents...)
	for index := range projected {
		if projected[index].Cache != nil {
			projected[index].Cache = nil
			(*dropped)++
		}
		if projected[index].ToolResult != nil {
			result := *projected[index].ToolResult
			result.Content = contentWithoutCacheDirectives(result.Content, dropped)
			projected[index].ToolResult = &result
		}
	}
	return projected
}

func projectChatCacheDirectives(request Request) (Request, Diagnostics) {
	dropped := 0
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if chatOmitsCacheDirective(content) {
				dropped++
			}
		}
	}
	if dropped == 0 {
		return request, nil
	}
	projected := request
	projected.Messages = append([]Message(nil), request.Messages...)
	for index := range projected.Messages {
		contents := append([]Content(nil), request.Messages[index].Content...)
		for position := range contents {
			if chatOmitsCacheDirective(contents[position]) {
				contents[position].Cache = nil
			}
		}
		projected.Messages[index].Content = contents
	}
	projected.Generation++ // The source envelope still contains cache_control.
	return projected, cacheDroppedDiagnostics(OpenAIChatV1, fmt.Sprintf(
		"Chat Completions cannot represent %d Anthropic prompt-cache boundaries on tool calls, reasoning, refusals or empty tool results", dropped))
}

// chatOmitsCacheDirective reports a boundary the Chat encoder has no place for.
// A tool result with text carries its boundary on its last text part.
func chatOmitsCacheDirective(content Content) bool {
	if content.Cache == nil {
		return false
	}
	switch content.Kind {
	case ContentToolCall, ContentReasoning, ContentRefusal:
		return true
	case ContentToolResult:
		return content.ToolResult != nil && len(content.ToolResult.Content) == 0
	default:
		return false
	}
}

func cacheDroppedDiagnostics(target WireFormat, reason string) Diagnostics {
	return Diagnostics{{
		Source: AnthropicMessagesV1,
		Target: target,
		Field:  "cache_control",
		Action: DiagnosticDropped,
		Reason: reason,
	}}
}
