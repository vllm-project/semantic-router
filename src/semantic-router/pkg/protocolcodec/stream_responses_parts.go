package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

type responsesContentPartSemantics struct {
	itemType  string
	scope     responsesContentScope
	kind      string
	eventType llmprotocol.EventType
	content   llmprotocol.Content
}

func interpretResponsesContentPart(part *responsesContentWire) (responsesContentPartSemantics, error) {
	if part == nil {
		return responsesContentPartSemantics{}, invalidProviderResponse("stream_content_part_required", "Responses content part event is incomplete")
	}
	result := responsesContentPartSemantics{
		itemType: "message", scope: responsesContentMessage, kind: part.Type,
		eventType: llmprotocol.EventOutputTextDelta,
	}
	switch part.Type {
	case "output_text":
		result.content = llmprotocol.Content{Kind: llmprotocol.ContentText, Text: part.Text}
	case "refusal":
		result.content = llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: part.Refusal}
	case "reasoning_text":
		result.itemType = "reasoning"
		result.scope = responsesContentReasoningText
		result.eventType = llmprotocol.EventReasoningDelta
		result.content = llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: part.Text, Reasoning: llmprotocol.ReasoningScopeText,
		}
	default:
		return responsesContentPartSemantics{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"unsupported_stream_content_part",
			"Responses stream content part is unsupported",
			nil,
		)
	}
	return result, nil
}
