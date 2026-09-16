package contextcompression

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func (request *RequestIR) initializeTransformMetadata(provenance Provenance) {
	request.attachOriginalHistory(provenance)
	turn := -1
	for _, message := range request.Messages {
		message.Source = messageSource(message, provenance)
		message.Protection = provenance.ProtectedMessages[message.Index]
		if message.Role == "system" || message.Role == "developer" {
			message.Protection |= ProtectInstructions
		}
		if request.messageStartsTurn(message) && message.Source == SourceHistory {
			turn = message.Index
		}
		message.TurnID = turn
		if request.messageHasNonText(message) {
			message.Protection |= ProtectMultimodal
		}
	}
	request.protectCurrentTurn(turn)
	request.labelToolExchanges()
	for _, message := range request.Messages {
		if message.Source == SourceHistory && message.Protection == 0 {
			message.Eligibility = EligibleHistoryRemoval
		}
	}
}

func messageSource(message *MessageIR, provenance Provenance) ContentSource {
	if _, ok := provenance.MemoryMessageIndexes[message.Index]; ok {
		return SourceMemory
	}
	rag, history := false, false
	ids := append(messageResultIDs(message), messageToolCallIDs(message)...)
	for _, id := range ids {
		if _, ok := provenance.RAGToolCallIDs[id]; ok && id != "" {
			rag = true
		} else {
			history = true
		}
	}
	for _, block := range message.Blocks {
		if block.ToolCallID == "" {
			history = true
		}
	}

	if rag && history {
		return SourceMixed
	}
	if rag {
		return SourceRAG
	}
	return SourceHistory
}

func (request *RequestIR) messageStartsTurn(message *MessageIR) bool {
	if message.Role != "user" {
		return false
	}
	// Anthropic represents tool results as user messages; these continue the
	// existing turn rather than opening a new one.
	if request.Semantic != nil {
		for _, content := range request.Semantic.Messages[message.Index].Content {
			if content.Kind != llmprotocol.ContentToolResult {
				return true
			}
		}
		return false
	}
	content, array := message.Raw["content"].([]interface{})
	if !array {
		return true
	}
	for _, raw := range content {
		block, ok := raw.(map[string]interface{})
		if !ok || block["type"] != "tool_result" {
			return true
		}
	}
	return false
}

func (request *RequestIR) messageHasNonText(message *MessageIR) bool {
	if request.Semantic != nil {
		return semanticHasProtectedBlocks(request.Semantic.Messages[message.Index].Content)
	}
	return rawHasProtectedBlocks(message.Raw["content"])
}

func semanticHasProtectedBlocks(blocks []llmprotocol.Content) bool {
	for _, block := range blocks {
		switch block.Kind {
		case llmprotocol.ContentText, llmprotocol.ContentToolCall:
		case llmprotocol.ContentToolResult:
			if block.ToolResult == nil || semanticHasProtectedBlocks(block.ToolResult.Content) {
				return true
			}
		default:
			return true
		}
	}
	return false
}

func rawHasProtectedBlocks(content interface{}) bool {
	blocks, ok := content.([]interface{})
	if !ok {
		return false
	}
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			return true
		}
		switch block["type"] {
		case "text", "input_text", "tool_use":
		case "tool_result":
			if rawHasProtectedBlocks(block["content"]) {
				return true
			}
		default:
			return true
		}
	}
	return false
}

func (request *RequestIR) protectCurrentTurn(turn int) {
	for _, message := range request.Messages {
		if turn >= 0 && message.TurnID == turn {
			message.Protection |= ProtectLiveTurn
		}
		if message.Protected && message.TurnID == turn {
			message.Protection |= ProtectContinuation
		}
		if message.TurnID < 0 {
			message.Protection |= ProtectUnknown
		}
	}
}

func (request *RequestIR) attachOriginalHistory(provenance Provenance) {
	request.originalHistory = provenance.OriginalHistory
	if request.originalHistory == nil && len(provenance.RAGToolCallIDs) == 0 && len(provenance.MemoryMessageIndexes) == 0 {
		request.originalHistory = CaptureHistory(request.Semantic)
	}
}
