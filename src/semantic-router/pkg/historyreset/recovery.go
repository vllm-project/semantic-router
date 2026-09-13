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

// envelopeWireMessage carries an already-encoded message so the payload is
// measured exactly, and only once, while it is being built.
type envelopeWireMessage struct {
	ID      int             `json:"id"`
	TurnID  int             `json:"turn_id"`
	Message json.RawMessage `json:"message"`
}

type envelopeWire struct {
	Version  string                `json:"version"`
	Removed  []envelopeWireMessage `json:"removed"`
	Turns    int                   `json:"turns"`
	Messages int                   `json:"messages"`
}

// errRecoveryPayloadTooLarge stops envelope construction as soon as the
// encoded content passes the configured bound.
var errRecoveryPayloadTooLarge = fmt.Errorf("removed history exceeds the recovery payload bound")

// buildEnvelope resolves the proposed IDs against the detached pre-transform
// messages. A missing binding is an error rather than a partial payload: the
// action must never remove history it cannot store completely.
// buildEnvelope resolves the proposed IDs against the detached pre-transform
// messages and encodes them under the supplied bound. Each message is encoded
// once and measured as it is added, so an oversized removal stops after the
// first message that crosses the limit instead of serializing the whole
// conversation and checking afterwards. Encoding is not interruptible, so the
// bound is enforced between messages rather than inside one.
//
// A missing binding is an error rather than a partial payload: the action must
// never remove history it cannot store completely.
func (a *Action) buildEnvelope(
	ids []int,
	turns map[int]int,
	limit int,
) (string, error) {
	wire := envelopeWire{Version: EnvelopeVersion, Removed: make([]envelopeWireMessage, 0, len(ids))}
	distinct := make(map[int]struct{}, len(ids))
	encoded := envelopeFixedOverhead + len(EnvelopeVersion)
	for _, id := range ids {
		message, ok := a.detached[id]
		if !ok {
			return "", fmt.Errorf("removed message %d has no detached content", id)
		}
		payload, err := json.Marshal(message)
		if err != nil {
			return "", fmt.Errorf("encode removed message %d: %w", id, err)
		}
		encoded += len(payload) + envelopeMessageOverhead
		if limit > 0 && encoded > limit {
			return "", errRecoveryPayloadTooLarge
		}
		wire.Removed = append(wire.Removed, envelopeWireMessage{
			ID: id, TurnID: turns[id], Message: payload,
		})
		distinct[turns[id]] = struct{}{}
	}
	wire.Turns, wire.Messages = len(distinct), len(wire.Removed)
	body, err := json.Marshal(wire)
	if err != nil {
		return "", fmt.Errorf("encode removed history: %w", err)
	}
	if limit > 0 && len(body) > limit {
		return "", errRecoveryPayloadTooLarge
	}
	return string(body), nil
}

// Envelope framing overheads counted alongside each encoded message so the
// running total stays an upper bound on the final document. The bias is
// deliberate: the bound may refuse a payload slightly under the limit, but it
// can never accept one above it.
const (
	envelopeFixedOverhead   = 96
	envelopeMessageOverhead = 64
)
