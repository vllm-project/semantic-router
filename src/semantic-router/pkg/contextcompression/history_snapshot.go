package contextcompression

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// HistorySnapshot owns an immutable copy captured before RAG and Memory. Its
// accessor returns detached semantic values, including nested tool/media blocks.
// The encoding is private storage of typed Go values, never a provider payload.
type HistorySnapshot struct {
	data []byte
}

type ConversationHistory struct {
	Instructions []llmprotocol.InstructionBlock
	Messages     []llmprotocol.Message
}

func CaptureHistory(request *llmprotocol.Request) *HistorySnapshot {
	if request == nil {
		return nil
	}
	data, err := json.Marshal(ConversationHistory{Instructions: request.Instructions, Messages: request.Messages})
	if err != nil {
		return nil
	}
	return &HistorySnapshot{data: data}
}

func (snapshot *HistorySnapshot) Conversation() ConversationHistory {
	var history ConversationHistory
	if snapshot != nil {
		// Only bytes produced by CaptureHistory are stored here.
		_ = json.Unmarshal(snapshot.data, &history)
	}
	return history
}

func (request *RequestIR) OriginalHistory() ConversationHistory {
	return request.originalHistory.Conversation()
}
