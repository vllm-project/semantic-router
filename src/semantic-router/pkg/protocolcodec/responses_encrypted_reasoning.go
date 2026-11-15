package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func responsesRequestsEncryptedReasoning(raw json.RawMessage) bool {
	var include []string
	_ = json.Unmarshal(raw, &include)
	return len(include) > 0
}

func decodeResponsesEncryptedReasoning(item responsesItemWire, policy llmprotocol.Policy, provider bool) (*llmprotocol.Content, error) {
	if len(item.EncryptedContent) == 0 || bytes.Equal(bytes.TrimSpace(item.EncryptedContent), []byte("null")) {
		return nil, nil
	}
	var data string
	if err := json.Unmarshal(item.EncryptedContent, &data); err != nil {
		return nil, responsesItemDecodeError(provider, "encrypted_content must be a string", err)
	}
	// Empty ciphertext carries no continuation state.
	if data == "" {
		return nil, nil
	}
	detail := &llmprotocol.ReasoningDetail{
		Type:   llmprotocol.ReasoningDetailEncrypted,
		Format: llmprotocol.OptionalString{Present: true, Value: llmprotocol.ReasoningFormatResponses},
		Data:   llmprotocol.OptionalString{Present: true, Value: data},
	}
	if item.ID != "" {
		detail.ID = llmprotocol.OptionalString{Present: true, Value: item.ID}
	}
	if err := llmprotocol.ValidateReasoningDetail(*detail, policy.Limits); err != nil {
		return nil, responsesItemDecodeError(provider, "encrypted reasoning exceeds the configured limits", err)
	}
	return &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, ReasoningDetail: detail}, nil
}
