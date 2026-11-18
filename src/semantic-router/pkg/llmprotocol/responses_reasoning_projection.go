package llmprotocol

// ProjectResponsesReasoning applies the destination's replay policy to a copy.
// Other continuity formats remain fail-closed at the capability/codec checks.
func ProjectResponsesReasoning(request Request, target WireFormat, openAIHosted bool) (Request, Diagnostics) {
	var diagnostics Diagnostics
	dropped := func(field, reason string) {
		diagnostics = append(diagnostics, Diagnostic{
			Source: request.Trusted.SourceFormat, Target: target,
			Field: field, Action: DiagnosticDropped, Reason: reason,
		})
	}
	if request.IncludeEncryptedReasoning && !openAIHosted {
		request.IncludeEncryptedReasoning = false
		dropped("include", "the destination does not receive Responses encrypted reasoning")
	}
	messages := make([]Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		content := make([]Content, 0, len(message.Content))
		for at := 0; at < len(message.Content); {
			block := message.Content[at]
			if block.Kind != ContentReasoning {
				content = append(content, block)
				at++
				continue
			}
			end := at + 1
			for end < len(message.Content) && message.Content[end].Kind == ContentReasoning {
				end++
			}
			hasBlob := false
			for _, part := range message.Content[at:end] {
				hasBlob = hasBlob || IsResponsesReasoning(part.ReasoningDetail)
			}
			for _, part := range message.Content[at:end] {
				switch {
				case IsResponsesReasoning(part.ReasoningDetail):
					if openAIHosted {
						content = append(content, part)
					} else {
						dropped("input[].encrypted_content", "Responses encrypted reasoning is incompatible with the destination")
					}
				case openAIHosted && part.ReasoningDetail == nil && part.Signature == "":
					if hasBlob && part.Reasoning == ReasoningScopeSummary {
						content = append(content, part)
					} else {
						dropped("input[].reasoning", "OpenAI-hosted Responses requires encrypted reasoning history")
					}
				default:
					content = append(content, part)
				}
			}
			at = end
		}
		if len(content) == len(message.Content) {
			messages = append(messages, message)
			continue
		}
		if len(content) > 0 {
			message.Content = content
			messages = append(messages, message)
		}
	}
	if len(diagnostics) == 0 {
		return request, nil
	}
	request.Messages = messages
	request.Generation++
	return request, diagnostics
}
