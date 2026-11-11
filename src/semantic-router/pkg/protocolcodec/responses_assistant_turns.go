package protocolcodec

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Order of content within one assistant turn: reasoning, then text, then calls.
const (
	turnRankReasoning = iota
	turnRankText
	turnRankCall
)

// responsesAssistantTurns returns the request's messages with adjacent
// Responses assistant items joined into the assistant turn they came from.
// Responses sends reasoning, the assistant message and each function or custom
// call as separate items. Chat and Messages backends render every message as
// its own turn, and a model shown turns that end in text without a call ends
// its own turn the same way. Items join only in turn order, so nothing moves
// across other content, and only at this projection boundary, so native
// Responses keeps its item identities.
func responsesAssistantTurns(request llmprotocol.Request) []llmprotocol.Message {
	if request.Trusted.SourceFormat != llmprotocol.OpenAIResponsesV1 {
		return request.Messages
	}
	turns := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		if n := len(turns); n > 0 && continuesAssistantTurn(turns[n-1], message) {
			// Clip so the append never writes into the request's own content.
			turns[n-1].Content = append(slices.Clip(turns[n-1].Content), message.Content...)
			continue
		}
		turns = append(turns, message)
	}
	return turns
}

func continuesAssistantTurn(previous, next llmprotocol.Message) bool {
	if previous.Role != llmprotocol.RoleAssistant || next.Role != llmprotocol.RoleAssistant ||
		previous.Name != next.Name || previous.ReasoningEffort != "" || next.ReasoningEffort != "" ||
		len(previous.Content) == 0 || len(next.Content) == 0 {
		return false
	}
	last, first := turnRank(previous.Content[len(previous.Content)-1]), turnRank(next.Content[0])
	return first > last || first == turnRankCall && last == turnRankCall
}

func turnRank(content llmprotocol.Content) int {
	switch content.Kind {
	case llmprotocol.ContentReasoning:
		return turnRankReasoning
	case llmprotocol.ContentToolCall:
		return turnRankCall
	default:
		return turnRankText
	}
}
