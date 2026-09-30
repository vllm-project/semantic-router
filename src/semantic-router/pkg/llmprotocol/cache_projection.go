package llmprotocol

import "fmt"

// ProjectAnthropicCacheDirectives omits Anthropic prompt-cache boundaries only
// when dispatching to Responses, which has no per-block cache directive. Cache
// controls are performance hints, not prompt content. The caller must surface
// the returned dropped diagnostic to the public client. Other source/target
// pairs retain the strict capability contract.
func ProjectAnthropicCacheDirectives(request Request, target WireFormat) (Request, Diagnostics) {
	if request.Trusted.SourceFormat != AnthropicMessagesV1 || target != OpenAIResponsesV1 ||
		!RequiredCapabilities(request).Supports(CapabilityCacheDirectives) {
		return request, nil
	}

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
	return projected, Diagnostics{{
		Source: AnthropicMessagesV1,
		Target: OpenAIResponsesV1,
		Field:  "cache_control",
		Action: DiagnosticDropped,
		Reason: fmt.Sprintf("Responses cannot represent %d Anthropic prompt-cache boundaries", dropped),
	}}
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
