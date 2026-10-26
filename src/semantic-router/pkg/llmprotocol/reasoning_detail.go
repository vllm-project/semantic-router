package llmprotocol

import (
	"encoding/json"
	"fmt"
	"unicode/utf8"
)

// OptionalString preserves absent, null and string values in continuity blocks.
// Presence matters when a stream updates metadata without a text delta.
type OptionalString struct {
	Value   string
	Present bool
	Null    bool
}

func (value OptionalString) IsZero() bool { return !value.Present }
func (value OptionalString) MarshalJSON() ([]byte, error) {
	if value.Null || !value.Present {
		return []byte("null"), nil
	}
	return json.Marshal(value.Value)
}

func (value *OptionalString) UnmarshalJSON(body []byte) error {
	var decoded *string
	if err := json.Unmarshal(body, &decoded); err != nil {
		return err
	}
	*value = OptionalString{Present: true, Null: decoded == nil}
	if decoded != nil {
		value.Value = *decoded
	}
	return nil
}

// ReasoningDetail preserves OpenRouter's closed, typed reasoning union. Unlike
// plain reasoning text these blocks may carry signed or encrypted continuation
// state that cannot be summarized or forwarded to an arbitrary model dialect.
type ReasoningDetail struct {
	Type      string         `json:"type"`
	ID        OptionalString `json:"id,omitzero"`
	Format    OptionalString `json:"format,omitzero"`
	Index     *int64         `json:"index,omitempty"`
	Text      OptionalString `json:"text,omitzero"`
	Summary   OptionalString `json:"summary,omitzero"`
	Data      OptionalString `json:"data,omitzero"`
	Signature OptionalString `json:"signature,omitzero"`
}

// ValidateReasoningDetail checks the closed variant and bounded payload.
func ValidateReasoningDetail(detail ReasoningDetail, limits Limits) error {
	invalid := func(message string) error {
		return NewError(ErrorInvalidRequest, "invalid_reasoning_detail", message, nil)
	}
	switch detail.Type {
	case "reasoning.text":
		if detail.Summary.Present || detail.Data.Present {
			return invalid("text reasoning contains fields from another variant")
		}
	case "reasoning.summary":
		if !detail.Summary.Present || detail.Summary.Null {
			return invalid("reasoning summary requires a string summary")
		}
		if detail.Text.Present || detail.Data.Present || detail.Signature.Present {
			return invalid("reasoning summary contains fields from another variant")
		}
	case "reasoning.encrypted":
		if !detail.Data.Present || detail.Data.Null {
			return invalid("encrypted reasoning requires a string payload")
		}
		if detail.Text.Present || detail.Summary.Present || detail.Signature.Present {
			return invalid("encrypted reasoning contains fields from another variant")
		}
	default:
		return invalid("reasoning detail type is unsupported")
	}
	if detail.Index != nil && *detail.Index < 0 {
		return invalid("reasoning detail index must be nonnegative")
	}
	for _, value := range []OptionalString{detail.ID, detail.Format, detail.Text, detail.Summary, detail.Data, detail.Signature} {
		if !utf8.ValidString(value.Value) || (!value.Present || value.Null) && value.Value != "" {
			return invalid("reasoning detail has an invalid optional string")
		}
	}
	if limits.IdentifierBytes > 0 && len(detail.ID.Value) > limits.IdentifierBytes {
		return invalid("reasoning detail ID exceeds the configured limit")
	}
	if limits.IdentifierBytes > 0 && len(detail.Format.Value) > limits.IdentifierBytes {
		return invalid("reasoning detail format exceeds the configured limit")
	}
	if limits.TextBytes > 0 && detail.PayloadBytes() > limits.TextBytes {
		return invalid("reasoning detail payload exceeds the configured limit")
	}
	return nil
}

// PayloadBytes includes opaque data and signatures as well as display text.
func (detail ReasoningDetail) PayloadBytes() int {
	return len(detail.Text.Value) + len(detail.Summary.Value) + len(detail.Data.Value) + len(detail.Signature.Value)
}

// MergeReasoningDetail combines deltas of one block without changing its
// identity. Text, opaque data, and signature fragments append in wire order.
func MergeReasoningDetail(current, delta ReasoningDetail) (ReasoningDetail, error) {
	if current.Type != delta.Type {
		return current, fmt.Errorf("reasoning detail changed type")
	}
	if delta.ID.Present {
		if current.ID.Present && !current.ID.Null && current.ID != delta.ID {
			return current, fmt.Errorf("reasoning detail changed ID")
		}
		current.ID = delta.ID
	}
	if delta.Format.Present {
		if current.Format.Present && !current.Format.Null && current.Format != delta.Format {
			return current, fmt.Errorf("reasoning detail changed format")
		}
		current.Format = delta.Format
	}
	if delta.Index != nil {
		if current.Index != nil && *current.Index != *delta.Index {
			return current, fmt.Errorf("reasoning detail changed index")
		}
		current.Index = delta.Index
	}
	for _, pair := range [][2]*OptionalString{{&current.Text, &delta.Text}, {&current.Summary, &delta.Summary}, {&current.Data, &delta.Data}, {&current.Signature, &delta.Signature}} {
		if !pair[1].Present {
			continue
		}
		if !pair[0].Present || pair[0].Null {
			*pair[0] = *pair[1]
		} else if !pair[1].Null {
			pair[0].Value += pair[1].Value
		}
	}
	return current, nil
}
