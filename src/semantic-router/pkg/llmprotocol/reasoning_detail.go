package llmprotocol

import (
	"fmt"
	"unicode/utf8"
)

// ReasoningDetailType names one variant of a typed reasoning continuity block.
type ReasoningDetailType string

const (
	ReasoningDetailText      ReasoningDetailType = "text"
	ReasoningDetailSummary   ReasoningDetailType = "summary"
	ReasoningDetailEncrypted ReasoningDetailType = "encrypted"
	// ReasoningDetailServerToolCall records a provider-hosted tool invocation
	// so that a later turn can rehydrate it.
	ReasoningDetailServerToolCall ReasoningDetailType = "server_tool_call"
)

// OptionalString distinguishes an absent value, an explicit null, and a
// string. Continuity blocks return to the provider exactly as received, and a
// stream can update metadata without a text fragment.
type OptionalString struct {
	Value   string
	Present bool
	Null    bool
}

// Set reports whether the value is a string rather than absent or null.
func (value OptionalString) Set() bool { return value.Present && !value.Null }

// ReasoningDetail is a typed reasoning continuity block. Unlike plain
// reasoning text it may carry signed or encrypted state that the provider needs
// back unchanged to continue a tool loop, so it is never summarized into
// display text or forwarded to a format that cannot carry it.
type ReasoningDetail struct {
	Type       ReasoningDetailType
	ID         OptionalString
	Format     OptionalString
	Index      *int64
	Text       OptionalString
	Signature  OptionalString
	Summary    OptionalString
	Data       OptionalString
	ToolName   OptionalString
	ToolCallID OptionalString
	Arguments  OptionalString
	Result     OptionalString
}

type reasoningDetailRule uint8

const (
	reasoningFieldForbidden reasoningDetailRule = iota
	reasoningFieldOptional
	reasoningFieldRequired
)

// reasoningDetailVariants closes the union. Each rule applies to the field at
// the same position in payloadFields; identity fields are shared by all
// variants.
var reasoningDetailVariants = map[ReasoningDetailType][8]reasoningDetailRule{
	// Text, Signature, Summary, Data, ToolName, ToolCallID, Arguments, Result
	ReasoningDetailText:      {reasoningFieldOptional, reasoningFieldOptional},
	ReasoningDetailSummary:   {2: reasoningFieldRequired},
	ReasoningDetailEncrypted: {3: reasoningFieldRequired},
	ReasoningDetailServerToolCall: {
		4: reasoningFieldRequired, 5: reasoningFieldOptional, 6: reasoningFieldRequired, 7: reasoningFieldRequired,
	},
}

func (detail *ReasoningDetail) payloadFields() [8]*OptionalString {
	return [8]*OptionalString{
		&detail.Text, &detail.Signature, &detail.Summary, &detail.Data,
		&detail.ToolName, &detail.ToolCallID, &detail.Arguments, &detail.Result,
	}
}

// ValidateReasoningDetail checks the closed variant and bounded payload.
func ValidateReasoningDetail(detail ReasoningDetail, limits Limits) error {
	invalid := func(message string) error {
		return NewError(ErrorInvalidRequest, "invalid_reasoning_detail", message, nil)
	}
	rules, known := reasoningDetailVariants[detail.Type]
	if !known {
		return invalid("reasoning detail type is unsupported")
	}
	payload := detail.payloadFields()
	for position, value := range payload {
		switch {
		case rules[position] == reasoningFieldForbidden && value.Present:
			return invalid("reasoning detail contains a field from another variant")
		case rules[position] == reasoningFieldRequired && !value.Set():
			return invalid("reasoning detail is missing a required string")
		}
	}
	if detail.Index != nil && *detail.Index < 0 {
		return invalid("reasoning detail index must be nonnegative")
	}
	for _, value := range append(payload[:], &detail.ID, &detail.Format) {
		if !utf8.ValidString(value.Value) || !value.Set() && value.Value != "" {
			return invalid("reasoning detail has an invalid optional string")
		}
	}
	for _, identifier := range []string{detail.ID.Value, detail.Format.Value, detail.ToolName.Value, detail.ToolCallID.Value} {
		if limits.IdentifierBytes > 0 && len(identifier) > limits.IdentifierBytes {
			return invalid("reasoning detail identifier exceeds the configured limit")
		}
	}
	if limits.TextBytes > 0 && detail.PayloadBytes() > limits.TextBytes {
		return invalid("reasoning detail payload exceeds the configured limit")
	}
	return nil
}

// PayloadBytes includes opaque data, signatures, and server tool records as
// well as display text.
func (detail ReasoningDetail) PayloadBytes() int {
	total := 0
	for _, value := range detail.payloadFields() {
		total += len(value.Value)
	}
	return total
}

// MergeReasoningDetail combines fragments of one block without changing its
// identity. Payload fragments append in wire order.
func MergeReasoningDetail(current, delta ReasoningDetail) (ReasoningDetail, error) {
	if current.Type != delta.Type {
		return current, fmt.Errorf("reasoning detail changed type")
	}
	identities := [][2]*OptionalString{
		{&current.ID, &delta.ID},
		{&current.Format, &delta.Format},
		{&current.ToolName, &delta.ToolName},
		{&current.ToolCallID, &delta.ToolCallID},
	}
	for _, pair := range identities {
		// A later null carries no identity, so it cannot unset an earlier one.
		if !pair[1].Present || pair[1].Null && pair[0].Set() {
			continue
		}
		if pair[0].Set() && *pair[0] != *pair[1] {
			return current, fmt.Errorf("reasoning detail changed identity")
		}
		*pair[0] = *pair[1]
	}
	if delta.Index != nil {
		if current.Index != nil && *current.Index != *delta.Index {
			return current, fmt.Errorf("reasoning detail changed index")
		}
		current.Index = delta.Index
	}
	payload := [][2]*OptionalString{
		{&current.Text, &delta.Text},
		{&current.Signature, &delta.Signature},
		{&current.Summary, &delta.Summary},
		{&current.Data, &delta.Data},
		{&current.Arguments, &delta.Arguments},
		{&current.Result, &delta.Result},
	}
	for _, pair := range payload {
		if !pair[1].Present {
			continue
		}
		if !pair[0].Set() {
			*pair[0] = *pair[1]
		} else if !pair[1].Null {
			pair[0].Value += pair[1].Value
		}
	}
	return current, nil
}
