package memory

// Provenance travels in the metadata blob of every store that keeps one
// (Milvus, Valkey). Writers and every parse path share these two helpers so a
// store cannot persist provenance on one path and drop it on another.

func putProvenanceMetadata(metadata map[string]interface{}, memory *Memory) {
	metadata["created_by_user_id"] = memory.CreatedByUserID
	metadata["conversation_id"] = memory.ConversationID
	metadata["created_via"] = memory.CreatedVia
}

func applyProvenanceMetadata(memory *Memory, metadata map[string]interface{}) {
	if createdByUserID, ok := metadata["created_by_user_id"].(string); ok {
		memory.CreatedByUserID = createdByUserID
	}
	if conversationID, ok := metadata["conversation_id"].(string); ok {
		memory.ConversationID = conversationID
	}
	if createdVia, ok := metadata["created_via"].(string); ok {
		memory.CreatedVia = createdVia
	}
}
