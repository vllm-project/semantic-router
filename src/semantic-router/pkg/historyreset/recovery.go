package historyreset

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"unicode/utf8"

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
// messages and encodes them under the supplied bound. Each message is measured
// in raw form, encoded once, and measured again exactly, so an oversized
// removal stops at the first message that crosses the limit instead of
// serializing the whole conversation and checking afterwards.
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
		// Encoding a message cannot be interrupted, and the neutral protocol
		// permits both very large single fields and very many small blocks.
		// Sizing the encoded form first keeps the work bounded: a message that
		// would not fit is refused before its encoded bytes are allocated.
		if limit > 0 && encoded+encodedMessageSize(message) > limit {
			return "", errRecoveryPayloadTooLarge
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

// encodedMessageSize reports how many bytes a message will occupy once
// encoded, without encoding it. Structure dominates for some requests: the
// neutral content block carries no omitempty tags, so a message of empty
// blocks encodes to tens of times its content length. Counting only the
// strings would let such a message pass a budget check and then allocate far
// beyond it, so the walk covers field names, delimiters, null pointers, and
// escaping exactly.
//
// Reflection keeps this honest across schema drift: a field added to any
// serialized type is counted without editing this file.
func encodedMessageSize(message llmprotocol.Message) int {
	return encodedValueSize(reflect.ValueOf(message))
}

func encodedValueSize(value reflect.Value) int {
	switch value.Kind() {
	case reflect.String:
		return encodedStringSize(value.String())
	case reflect.Bool:
		if value.Bool() {
			return len("true")
		}
		return len("false")
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return len(strconv.FormatInt(value.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return len(strconv.FormatUint(value.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		return len(strconv.FormatFloat(value.Float(), 'g', -1, 64))
	case reflect.Pointer, reflect.Interface:
		if value.IsNil() {
			return len("null")
		}
		return encodedValueSize(value.Elem())
	case reflect.Slice, reflect.Array:
		return encodedSequenceSize(value)
	case reflect.Struct:
		return encodedStructSize(value)
	default:
		// An unexpected kind must not silently count as nothing. Charge the
		// largest plausible scalar encoding so the bound stays conservative.
		return unknownKindSize
	}
}

func encodedSequenceSize(value reflect.Value) int {
	if value.Kind() == reflect.Slice && value.IsNil() {
		return len("null")
	}
	total := len("[]")
	for index := 0; index < value.Len(); index++ {
		if index > 0 {
			total++ // comma
		}
		total += encodedValueSize(value.Index(index))
	}
	return total
}

func encodedStructSize(value reflect.Value) int {
	total := len("{}")
	fields := 0
	structType := value.Type()
	for index := 0; index < structType.NumField(); index++ {
		field := structType.Field(index)
		if !field.IsExported() {
			continue
		}
		if fields > 0 {
			total++ // comma
		}
		fields++
		// "Name": plus the encoded value.
		total += len(field.Name) + len(`"":`) + encodedValueSize(value.Field(index))
	}
	return total
}

// encodedStringSize counts a JSON string exactly, including the quotes and the
// escapes the standard encoder emits: control characters and the HTML-sensitive
// characters expand to six bytes, and invalid UTF-8 becomes a replacement
// escape.
func encodedStringSize(value string) int {
	total := len(`""`)
	for index := 0; index < len(value); {
		character := value[index]
		if character < utf8.RuneSelf {
			switch {
			case character == '"' || character == '\\' ||
				character == '\n' || character == '\r' || character == '\t':
				total += 2
			case character < 0x20 || character == '<' || character == '>' || character == '&':
				total += 6
			default:
				total++
			}
			index++
			continue
		}
		decoded, size := utf8.DecodeRuneInString(value[index:])
		switch {
		case decoded == utf8.RuneError && size == 1:
			total += 6
			index++
		case decoded == '\u2028' || decoded == '\u2029':
			total += 6
			index += size
		default:
			total += size
			index += size
		}
	}
	return total
}

// unknownKindSize is charged for a kind this walk does not model, so a future
// field shape cannot be counted as free.
const unknownKindSize = 64
