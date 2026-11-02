package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chatOptionalString keeps a continuity field absent, null, or a string on
// the wire, so a block returns to the provider exactly as received.
type chatOptionalString llmprotocol.OptionalString

func (value chatOptionalString) IsZero() bool { return !value.Present }

func (value chatOptionalString) MarshalJSON() ([]byte, error) {
	if !llmprotocol.OptionalString(value).Set() {
		return []byte("null"), nil
	}
	return json.Marshal(value.Value)
}

func (value *chatOptionalString) UnmarshalJSON(body []byte) error {
	var decoded *string
	if err := json.Unmarshal(body, &decoded); err != nil {
		return err
	}
	*value = chatOptionalString{Present: true, Null: decoded == nil}
	if decoded != nil {
		value.Value = *decoded
	}
	return nil
}

// chatReasoningDetailWire is one OpenRouter reasoning_details entry, on
// assistant messages and stream deltas alike.
type chatReasoningDetailWire struct {
	Type       string             `json:"type"`
	ID         chatOptionalString `json:"id,omitzero"`
	Format     chatOptionalString `json:"format,omitzero"`
	Index      *int64             `json:"index,omitempty"`
	Text       chatOptionalString `json:"text,omitzero"`
	Signature  chatOptionalString `json:"signature,omitzero"`
	Summary    chatOptionalString `json:"summary,omitzero"`
	Data       chatOptionalString `json:"data,omitzero"`
	ToolName   chatOptionalString `json:"tool_name,omitzero"`
	ToolCallID chatOptionalString `json:"tool_call_id,omitzero"`
	Arguments  chatOptionalString `json:"arguments,omitzero"`
	Result     chatOptionalString `json:"result,omitzero"`
}

var chatReasoningDetailTypes = []struct {
	wire    string
	neutral llmprotocol.ReasoningDetailType
}{
	{"reasoning.text", llmprotocol.ReasoningDetailText},
	{"reasoning.summary", llmprotocol.ReasoningDetailSummary},
	{"reasoning.encrypted", llmprotocol.ReasoningDetailEncrypted},
	{"reasoning.server_tool_call", llmprotocol.ReasoningDetailServerToolCall},
}

// decodeChatReasoningDetail returns a client-side error; upstream callers map
// it to a provider failure.
func decodeChatReasoningDetail(wire chatReasoningDetailWire, limits llmprotocol.Limits) (llmprotocol.ReasoningDetail, error) {
	detail := llmprotocol.ReasoningDetail{
		ID: llmprotocol.OptionalString(wire.ID), Format: llmprotocol.OptionalString(wire.Format), Index: wire.Index,
		Text: llmprotocol.OptionalString(wire.Text), Signature: llmprotocol.OptionalString(wire.Signature),
		Summary: llmprotocol.OptionalString(wire.Summary), Data: llmprotocol.OptionalString(wire.Data),
		ToolName: llmprotocol.OptionalString(wire.ToolName), ToolCallID: llmprotocol.OptionalString(wire.ToolCallID),
		Arguments: llmprotocol.OptionalString(wire.Arguments), Result: llmprotocol.OptionalString(wire.Result),
	}
	for _, variant := range chatReasoningDetailTypes {
		if variant.wire == wire.Type {
			detail.Type = variant.neutral
		}
	}
	return detail, llmprotocol.ValidateReasoningDetail(detail, limits)
}

func encodeChatReasoningDetail(detail llmprotocol.ReasoningDetail) chatReasoningDetailWire {
	wire := chatReasoningDetailWire{
		ID: chatOptionalString(detail.ID), Format: chatOptionalString(detail.Format), Index: detail.Index,
		Text: chatOptionalString(detail.Text), Signature: chatOptionalString(detail.Signature),
		Summary: chatOptionalString(detail.Summary), Data: chatOptionalString(detail.Data),
		ToolName: chatOptionalString(detail.ToolName), ToolCallID: chatOptionalString(detail.ToolCallID),
		Arguments: chatOptionalString(detail.Arguments), Result: chatOptionalString(detail.Result),
	}
	for _, variant := range chatReasoningDetailTypes {
		if variant.neutral == detail.Type {
			wire.Type = variant.wire
		}
	}
	return wire
}

// decodeChatReasoningDetails converts the continuity blocks of one message.
// Only an assistant message can carry them.
func decodeChatReasoningDetails(wires []chatReasoningDetailWire, role llmprotocol.Role, limits llmprotocol.Limits) ([]llmprotocol.Content, error) {
	if len(wires) == 0 {
		return nil, nil
	}
	if role != llmprotocol.RoleAssistant {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_reasoning_detail_role", "reasoning details require an assistant message", nil)
	}
	contents := make([]llmprotocol.Content, 0, len(wires))
	for _, wire := range wires {
		detail, err := decodeChatReasoningDetail(wire, limits)
		if err != nil {
			return nil, err
		}
		contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, ReasoningDetail: &detail})
	}
	return contents, nil
}

func chatChoicesHaveReasoningDetails(choices []chatChoiceWire) bool {
	for _, choice := range choices {
		if len(choice.Message.ReasoningDetails) > 0 {
			return true
		}
	}
	return false
}

type chatReasoningDetailBlock struct {
	contentIndex int
	detail       llmprotocol.ReasoningDetail
}

// A block is identified by its type together with its index or ID. Gemini
// models send text fragments and then an encrypted signature under the same
// index, so an index alone does not name one block.
type chatReasoningDetailIndexKey struct {
	detailType llmprotocol.ReasoningDetailType
	index      int64
}

type chatReasoningDetailIDKey struct {
	detailType llmprotocol.ReasoningDetailType
	id         string
}

// Each provider block owns a distinct neutral content index. Block identity is
// independent from text/summary scope and cannot share Chat's single text slot.
type chatReasoningDetailState struct {
	blocks  []*chatReasoningDetailBlock
	byIndex map[chatReasoningDetailIndexKey]*chatReasoningDetailBlock
	byID    map[chatReasoningDetailIDKey]*chatReasoningDetailBlock
}

func (decoder *chatStreamDecoder) decodeReasoningDetails(choice chatChunkChoiceWire) ([]llmprotocol.Event, error) {
	if len(choice.Delta.ReasoningDetails) == 0 {
		return nil, nil
	}
	if decoder.reasoningDetails == nil {
		decoder.reasoningDetails = &chatReasoningDetailState{
			byIndex: make(map[chatReasoningDetailIndexKey]*chatReasoningDetailBlock),
			byID:    make(map[chatReasoningDetailIDKey]*chatReasoningDetailBlock),
		}
	}
	state := decoder.reasoningDetails
	events := make([]llmprotocol.Event, 0, len(choice.Delta.ReasoningDetails))
	for _, wire := range choice.Delta.ReasoningDetails {
		detail, err := decodeChatReasoningDetail(wire, decoder.policy.Limits)
		if err != nil {
			return nil, invalidProviderResponse("invalid_reasoning_detail", "Upstream reasoning detail is invalid")
		}
		block, err := state.observe(detail, choice.Index, decoder)
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

func (state *chatReasoningDetailState) observe(detail llmprotocol.ReasoningDetail, itemIndex int, decoder *chatStreamDecoder) (*chatReasoningDetailBlock, error) {
	var block *chatReasoningDetailBlock
	if detail.Index != nil {
		block = state.byIndex[chatReasoningDetailIndexKey{detail.Type, *detail.Index}]
	}
	if detail.ID.Set() && detail.ID.Value != "" {
		byID := state.byID[chatReasoningDetailIDKey{detail.Type, detail.ID.Value}]
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
		block = &chatReasoningDetailBlock{contentIndex: decoder.nextContentIndexes[itemIndex], detail: detail}
		decoder.nextContentIndexes[itemIndex]++
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
		state.byIndex[chatReasoningDetailIndexKey{detail.Type, *detail.Index}] = block
	}
	if detail.ID.Set() && detail.ID.Value != "" {
		state.byID[chatReasoningDetailIDKey{detail.Type, detail.ID.Value}] = block
	}
	return block, nil
}
