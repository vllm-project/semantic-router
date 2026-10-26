package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func rejectReasoningContinuity(format llmprotocol.WireFormat, required llmprotocol.CapabilitySet) error {
	if !required.Supports(llmprotocol.CapabilityReasoningContinuity) {
		return nil
	}
	return llmprotocol.RequireCapabilities(format, llmprotocol.Capabilities(), llmprotocol.Capabilities(llmprotocol.CapabilityReasoningContinuity))
}

type chatReasoningDetailBlock struct {
	contentIndex int
	detail       llmprotocol.ReasoningDetail
}

func chatChoicesHaveReasoningDetails(choices []chatChoiceWire) bool {
	for _, choice := range choices {
		if len(choice.Message.ReasoningDetails) > 0 {
			return true
		}
	}
	return false
}

// Each provider block owns a distinct neutral content index. Block identity is
// independent from text/summary scope and cannot share Chat's single text slot.
type chatReasoningDetailState struct {
	blocks  []*chatReasoningDetailBlock
	byIndex map[int64]*chatReasoningDetailBlock
	byID    map[string]*chatReasoningDetailBlock
}

func (decoder *chatStreamDecoder) decodeReasoningDetails(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if decoder.reasoningDetails == nil {
		decoder.reasoningDetails = &chatReasoningDetailState{byIndex: make(map[int64]*chatReasoningDetailBlock), byID: make(map[string]*chatReasoningDetailBlock)}
	}
	state := decoder.reasoningDetails
	events := make([]llmprotocol.Event, 0, len(choice.Delta.ReasoningDetails))
	for _, detail := range choice.Delta.ReasoningDetails {
		if err := llmprotocol.ValidateReasoningDetail(detail, decoder.policy.Limits); err != nil {
			return nil, invalidProviderResponse("invalid_reasoning_detail", "Upstream reasoning detail is invalid")
		}
		block, err := state.observe(detail, decoder)
		if err != nil {
			return nil, err
		}
		decoder.lastWasReasoning = true
		content := llmprotocol.Content{Kind: llmprotocol.ContentReasoning, ReasoningDetail: &detail}
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemIndex: choice.Index, ContentIndex: block.contentIndex, Content: &content})
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	return events, nil
}

func (state *chatReasoningDetailState) observe(detail llmprotocol.ReasoningDetail, decoder *chatStreamDecoder) (*chatReasoningDetailBlock, error) {
	var block *chatReasoningDetailBlock
	if detail.Index != nil {
		block = state.byIndex[*detail.Index]
	}
	if detail.ID.Present && !detail.ID.Null && detail.ID.Value != "" {
		byID := state.byID[detail.ID.Value]
		if block != nil && byID != nil && block != byID {
			return nil, invalidProviderResponse("reasoning_detail_identity_mismatch", "Upstream reasoning detail identity is inconsistent")
		}
		if block == nil {
			block = byID
		}
	}
	// Without an index or ID there is no evidence that two fragments belong
	// to the same block. Keep anonymous blocks separately in arrival order.
	if block == nil {
		if limit := decoder.policy.Limits.ContentBlocks; limit > 0 && len(state.blocks) >= limit {
			return nil, invalidProviderResponse("content_blocks_limit", "Upstream reasoning detail block limit exceeded")
		}
		block = &chatReasoningDetailBlock{contentIndex: decoder.nextContentIndexes[0], detail: detail}
		decoder.nextContentIndexes[0]++
		state.blocks = append(state.blocks, block)
	} else {
		merged, err := llmprotocol.MergeReasoningDetail(block.detail, detail)
		if err != nil {
			return nil, invalidProviderResponse("reasoning_detail_identity_mismatch", "Upstream reasoning detail changed identity or type")
		}
		block.detail = merged
	}
	if err := llmprotocol.ValidateReasoningDetail(block.detail, decoder.policy.Limits); err != nil {
		return nil, invalidProviderResponse("reasoning_detail_limit", "Upstream reasoning detail exceeds the configured limit")
	}
	if detail.Index != nil {
		state.byIndex[*detail.Index] = block
	}
	if detail.ID.Present && !detail.ID.Null && detail.ID.Value != "" {
		state.byID[detail.ID.Value] = block
	}
	return block, nil
}
