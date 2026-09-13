package historyreset

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// EnvelopeVersion identifies the stored removed-history format. A reader that
// does not recognize the version must refuse the payload rather than guess.
const EnvelopeVersion = "vsr.history-reset.v1"

// EnvelopeMessage is one removed message with the identities needed to
// reconstruct the conversation: its stable request ID, the turn it belonged
// to, and the complete neutral message including tool calls and results.
type EnvelopeMessage struct {
	ID      int                 `json:"id"`
	TurnID  int                 `json:"turn_id"`
	Message llmprotocol.Message `json:"message"`
}

// Envelope carries the turns one reset removed, in their original order.
type Envelope struct {
	Version  string            `json:"version"`
	Removed  []EnvelopeMessage `json:"removed"`
	Turns    int               `json:"turns"`
	Messages int               `json:"messages"`
}

// RecoveryWriter persists a removed-history payload and returns the key the
// model may later present to retrieve it. The policy package never sees the
// store, its scope, or its credentials.
type RecoveryWriter interface {
	Store(ctx context.Context, payload string) (string, error)
}

// NewEnvelope builds the versioned payload for a set of removed messages. The
// caller supplies them in request order; this function does not reorder them.
func NewEnvelope(removed []EnvelopeMessage) Envelope {
	turns := make(map[int]struct{}, len(removed))
	for _, message := range removed {
		turns[message.TurnID] = struct{}{}
	}
	return Envelope{
		Version:  EnvelopeVersion,
		Removed:  removed,
		Turns:    len(turns),
		Messages: len(removed),
	}
}

func (envelope Envelope) Encode() (string, error) {
	payload, err := json.Marshal(envelope)
	if err != nil {
		return "", fmt.Errorf("encode removed history: %w", err)
	}
	return string(payload), nil
}

// DecodeEnvelope reads a stored payload and rejects an unknown version.
func DecodeEnvelope(payload string) (Envelope, error) {
	var envelope Envelope
	if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
		return Envelope{}, fmt.Errorf("decode removed history: %w", err)
	}
	if envelope.Version != EnvelopeVersion {
		return Envelope{}, fmt.Errorf("unsupported removed-history version %q", envelope.Version)
	}
	return envelope, nil
}

// estimateEnvelopeBytes approximates the stored payload before it is built.
// Serialization is not interruptible once started, so an oversized removal is
// rejected from this estimate rather than by cancelling a marshal in flight.
func (a *Action) estimateEnvelopeBytes(ids []int) int {
	total := len(EnvelopeVersion) + envelopeFixedOverhead
	for _, id := range ids {
		message, ok := a.detached[id]
		if !ok {
			continue
		}
		total += envelopeMessageOverhead + len(message.Role)
		for _, content := range message.Content {
			total += len(content.Text) + len(content.Kind) + len(content.MediaType) +
				len(content.URL) + len(content.Data)
			if content.ToolCall != nil {
				total += len(content.ToolCall.ID) + len(content.ToolCall.Name) +
					len(content.ToolCall.Arguments)
			}
			if content.ToolResult != nil {
				total += len(content.ToolResult.CallID)
				for _, nested := range content.ToolResult.Content {
					total += len(nested.Text) + len(nested.Data)
				}
			}
		}
	}
	return total
}

// Envelope encoding overheads. They only need to be the right order of
// magnitude: the exact payload is still measured before it is stored.
const (
	envelopeFixedOverhead   = 64
	envelopeMessageOverhead = 96
)

// buildEnvelope resolves the proposed IDs against the detached pre-transform
// messages. A missing binding is an error rather than a partial payload: the
// action must never remove history it cannot store completely.
func (a *Action) buildEnvelope(
	ids []int,
	turns map[int]int,
) (string, error) {
	removed := make([]EnvelopeMessage, 0, len(ids))
	for _, id := range ids {
		message, ok := a.detached[id]
		if !ok {
			return "", fmt.Errorf("removed message %d has no detached content", id)
		}
		removed = append(removed, EnvelopeMessage{ID: id, TurnID: turns[id], Message: message})
	}
	return NewEnvelope(removed).Encode()
}
