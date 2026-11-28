package llmprotocol

import "fmt"

// ProjectReasoningSignatures sends the signed reasoning in a request's history
// to a backend whose wire format cannot carry the signature: each block keeps
// its text and loses its signature. A Messages client sends back the thinking
// an Anthropic backend signed, so without this a session that moves to another
// backend fails on every later call. A Messages target keeps the signatures.
// The caller applies it only when the operator lists
// LossyFieldReasoningSignature and the target lacks
// CapabilityReasoningSignature, and must report the returned diagnostic.
func ProjectReasoningSignatures(request Request, target WireFormat) (Request, Diagnostics) {
	if target == AnthropicMessagesV1 {
		return request, nil
	}
	var messages []Message
	for i, message := range request.Messages {
		content, unsigned := withoutReasoningSignatures(message.Content)
		if !unsigned {
			continue
		}
		if messages == nil {
			messages = append([]Message(nil), request.Messages...)
		}
		messages[i].Content = content
	}
	if messages == nil {
		return request, nil
	}
	projected := request
	projected.Messages = messages
	projected.Generation++ // The source envelope still carries the signatures.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  LossyFieldReasoningSignature,
		Action: DiagnosticApproximated,
		Reason: "the selected backend cannot accept a reasoning signature; the reasoning text is kept",
	}}
}

// withoutReasoningSignatures copies contents without their reasoning
// signatures, those inside tool results included, and reports whether there
// were any. It never edits the caller's slices.
func withoutReasoningSignatures(contents []Content) ([]Content, bool) {
	var out []Content
	for i, content := range contents {
		switch {
		case content.Kind == ContentReasoning && content.Reasoning == ReasoningScopeRedacted:
			// There is no text-only rendering of an opaque redacted block.
			// Leave it intact so the capability check still refuses it, even
			// when the operator opted into lossy signature projection.
			continue
		case content.Kind == ContentReasoning && content.Signature != "":
			content.Signature = ""
		case content.Kind == ContentToolResult && content.ToolResult != nil:
			nested, unsigned := withoutReasoningSignatures(content.ToolResult.Content)
			if !unsigned {
				continue
			}
			result := *content.ToolResult
			result.Content = nested
			content.ToolResult = &result
		default:
			continue
		}
		if out == nil {
			out = append([]Content(nil), contents...)
		}
		out[i] = content
	}
	return out, out != nil
}

// RefuseReasoningSignatures refuses a request whose history holds signed
// reasoning for a backend that cannot accept the signature, naming the first
// signed block as the client sent it. It returns nil for a request without one.
func RefuseReasoningSignatures(request Request) error {
	field, signed := firstReasoningSignature(request.Messages)
	if !signed {
		return nil
	}
	message := "the selected model cannot accept signed thinking in the conversation"
	if field != "" {
		message = field + ": the selected model cannot accept signed thinking"
	}
	return NewError(ErrorUnsupportedFeature, "unsupported_capability", message+OptInHint(LossyFieldReasoningSignature), nil)
}

// firstReasoningSignature names the first signed reasoning block of messages
// as a Messages client sent it, such as messages[1].content[0].signature. The
// name is empty when the client's position of the block is unknown.
func firstReasoningSignature(messages []Message) (string, bool) {
	for at, message := range messages {
		for index, content := range message.Content {
			path, signed := reasoningSignaturePath(content)
			if !signed {
				continue
			}
			if message.SourcePosition == 0 {
				return "", true
			}
			// Decoding splits a client message at its tool results, so the
			// block's index counts the content of the earlier parts as well.
			block := index
			for _, earlier := range messages[:at] {
				if earlier.SourcePosition == message.SourcePosition {
					block += len(earlier.Content)
				}
			}
			return fmt.Sprintf("messages[%d].content[%d]%s", message.SourcePosition-1, block, path), true
		}
	}
	return "", false
}

func reasoningSignaturePath(content Content) (string, bool) {
	if content.Kind == ContentReasoning && content.Signature != "" {
		return ".signature", true
	}
	if content.Kind != ContentToolResult || content.ToolResult == nil {
		return "", false
	}
	for index, nested := range content.ToolResult.Content {
		if path, signed := reasoningSignaturePath(nested); signed {
			return fmt.Sprintf(".content[%d]%s", index, path), true
		}
	}
	return "", false
}
