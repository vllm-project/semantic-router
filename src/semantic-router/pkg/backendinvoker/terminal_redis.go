package backendinvoker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	responseTerminalEnvelopeSchema  = "vllm-sr.response-terminal.v1"
	managedTerminalShardCount       = 64
	defaultManagedTerminalCapacity  = 8_192
	maximumManagedTerminalCapacity  = 65_536
	maximumTerminalPayloadBytes     = 16 << 10
	defaultTerminalOperationTimeout = 2 * time.Second
	maximumTerminalOperationTimeout = 10 * time.Second
	maximumTerminalTTL              = time.Hour
)

var finalizeResponseTerminalScript = redis.NewScript(`
local clock = redis.call("TIME")
local now_ms = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", now_ms)
if redis.call("EXISTS", KEYS[1]) == 1 then
  return 2
end
if redis.call("ZCARD", KEYS[2]) >= tonumber(ARGV[3]) then
  return 3
end
local expires_at = now_ms + tonumber(ARGV[2])
local created = redis.call("SET", KEYS[1], ARGV[1], "PX", ARGV[2], "NX")
if not created then
  return 2
end
redis.call("ZADD", KEYS[2], expires_at, KEYS[1])
redis.call("PEXPIRE", KEYS[2], tonumber(ARGV[2]) + 60000)
return 1
`)

var takeResponseTerminalScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
if value then
  redis.call("DEL", KEYS[1])
end
redis.call("ZREM", KEYS[2], KEYS[1])
return value
`)

type RedisResponseTerminalStoreOptions struct {
	Client           redis.UniversalClient
	KeyPrefix        string
	TTL              time.Duration
	OperationTimeout time.Duration
	Capacity         int
}

// RedisResponseTerminalStore is the managed cross-replica terminal
// rendezvous. PostgreSQL remains the durable usage ledger; this bounded,
// expiring value exists only long enough to transfer one neutral terminal from
// the private dispatch replica to the owning ExtProc replica.
type RedisResponseTerminalStore struct {
	client           redis.UniversalClient
	keyNamespace     string
	ttl              time.Duration
	operationTimeout time.Duration
	perShardCapacity int
}

var _ ResponseTerminalStore = (*RedisResponseTerminalStore)(nil)

func NewRedisResponseTerminalStore(
	options RedisResponseTerminalStoreOptions,
) (*RedisResponseTerminalStore, error) {
	if options.Client == nil {
		return nil, fmt.Errorf("%w: Valkey client is required", ErrResponseTerminalInvalid)
	}
	prefix := strings.TrimSpace(options.KeyPrefix)
	if prefix == "" || prefix != options.KeyPrefix || strings.ContainsAny(prefix, "\r\n\t ") {
		return nil, fmt.Errorf("%w: Valkey key prefix is invalid", ErrResponseTerminalInvalid)
	}
	ttl := options.TTL
	if ttl == 0 {
		ttl = defaultTerminalTTL
	}
	if ttl < time.Second || ttl > maximumTerminalTTL {
		return nil, fmt.Errorf("%w: terminal TTL must be between one second and one hour", ErrResponseTerminalInvalid)
	}
	operationTimeout := options.OperationTimeout
	if operationTimeout == 0 {
		operationTimeout = defaultTerminalOperationTimeout
	}
	if operationTimeout < time.Millisecond || operationTimeout > maximumTerminalOperationTimeout {
		return nil, fmt.Errorf("%w: terminal operation timeout is invalid", ErrResponseTerminalInvalid)
	}
	capacity := options.Capacity
	if capacity == 0 {
		capacity = defaultManagedTerminalCapacity
	}
	if capacity < managedTerminalShardCount || capacity > maximumManagedTerminalCapacity ||
		capacity%managedTerminalShardCount != 0 {
		return nil, fmt.Errorf(
			"%w: terminal capacity must be a multiple of %d between %d and %d",
			ErrResponseTerminalInvalid, managedTerminalShardCount,
			managedTerminalShardCount, maximumManagedTerminalCapacity,
		)
	}
	return &RedisResponseTerminalStore{
		client: options.Client, keyNamespace: responseTerminalKeyNamespace(prefix), ttl: ttl,
		operationTimeout: operationTimeout,
		perShardCapacity: capacity / managedTerminalShardCount,
	}, nil
}

func responseTerminalKeyNamespace(prefix string) string {
	digest := sha256.Sum256([]byte(prefix))
	return hex.EncodeToString(digest[:16])
}

func (store *RedisResponseTerminalStore) Finalize(
	ctx context.Context,
	plan Plan,
	attempt AttemptResult,
	terminal ResponseTerminal,
) error {
	if store == nil || store.client == nil {
		return ErrResponseTerminalUnavailable
	}
	reference, err := ResponseTerminalReferenceFromPlan(plan)
	if err != nil {
		return err
	}
	if validationErr := validateResponseTerminalRecord(reference, attempt, terminal); validationErr != nil {
		return validationErr
	}
	record := ResponseTerminalRecord{
		Reference: reference, Attempt: attempt, Terminal: cloneResponseTerminal(terminal),
	}
	payload, err := encodeResponseTerminalRecord(record)
	if err != nil {
		return err
	}
	keys, err := store.keys(reference)
	if err != nil {
		return err
	}
	operationContext, cancel := store.operationContext(ctx)
	defer cancel()
	result, err := finalizeResponseTerminalScript.Run(
		operationContext,
		store.client,
		[]string{keys.record, keys.expiry},
		payload,
		store.ttl.Milliseconds(),
		store.perShardCapacity,
	).Int64()
	if err != nil {
		return fmt.Errorf("%w: write terminal evidence", ErrResponseTerminalUnavailable)
	}
	switch result {
	case 1:
		return nil
	case 2:
		return ErrResponseTerminalDuplicate
	case 3:
		return ErrResponseTerminalCapacity
	default:
		return fmt.Errorf("%w: terminal write returned an invalid result", ErrResponseTerminalUnavailable)
	}
}

func (store *RedisResponseTerminalStore) Take(
	ctx context.Context,
	reference ResponseTerminalReference,
) (ResponseTerminalRecord, bool, error) {
	if store == nil || store.client == nil {
		return ResponseTerminalRecord{}, false, ErrResponseTerminalUnavailable
	}
	keys, err := store.keys(reference)
	if err != nil {
		return ResponseTerminalRecord{}, false, err
	}
	operationContext, cancel := store.operationContext(ctx)
	defer cancel()
	payloadText, err := takeResponseTerminalScript.Run(
		operationContext, store.client, []string{keys.record, keys.expiry},
	).Text()
	if errors.Is(err, redis.Nil) {
		return ResponseTerminalRecord{}, false, nil
	}
	if err != nil {
		return ResponseTerminalRecord{}, false, fmt.Errorf("%w: consume terminal evidence", ErrResponseTerminalUnavailable)
	}
	payload := []byte(payloadText)
	record, err := decodeResponseTerminalRecord(payload)
	if err != nil {
		return ResponseTerminalRecord{}, false, err
	}
	if record.Reference != reference {
		return ResponseTerminalRecord{}, false, fmt.Errorf(
			"%w: stored terminal reference mismatch", ErrResponseTerminalUnavailable,
		)
	}
	return record, true, nil
}

type responseTerminalRedisKeys struct {
	record string
	expiry string
}

func (store *RedisResponseTerminalStore) keys(
	reference ResponseTerminalReference,
) (responseTerminalRedisKeys, error) {
	digest, err := reference.digest()
	if err != nil {
		return responseTerminalRedisKeys{}, err
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil || len(decoded) != sha256.Size {
		return responseTerminalRedisKeys{}, fmt.Errorf("%w: terminal digest is invalid", ErrResponseTerminalInvalid)
	}
	shard := int(decoded[0]) % managedTerminalShardCount
	base := fmt.Sprintf(
		"vsr:response-terminal:%s:{rt-%s-%02d}:v1",
		store.keyNamespace, store.keyNamespace[:8], shard,
	)
	return responseTerminalRedisKeys{
		record: base + ":record:" + digest,
		expiry: base + ":expiry",
	}, nil
}

func (store *RedisResponseTerminalStore) operationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= store.operationTimeout {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, store.operationTimeout)
}

type responseTerminalEnvelope struct {
	Schema    string                    `json:"schema"`
	Reference ResponseTerminalReference `json:"reference"`
	Attempt   responseTerminalAttempt   `json:"attempt"`
	Terminal  responseTerminalPayload   `json:"terminal"`
}

type responseTerminalAttempt struct {
	ID              string          `json:"id"`
	Number          int             `json:"number"`
	BackendID       string          `json:"backendId"`
	StartedAt       int64           `json:"startedAtUnixNano,omitempty"`
	State           AttemptState    `json:"state"`
	StatusCode      int             `json:"statusCode"`
	CompletedAt     int64           `json:"completedAtUnixNano,omitempty"`
	ErrorCode       string          `json:"errorCode,omitempty"`
	FallbackTrigger FallbackTrigger `json:"fallbackTrigger,omitempty"`
}

type responseTerminalPayload struct {
	Usage      llmprotocol.Usage      `json:"usage"`
	StopReason llmprotocol.StopReason `json:"stopReason"`
	Error      *responseTerminalError `json:"error,omitempty"`
}

type responseTerminalError struct {
	Category   llmprotocol.ErrorCategory `json:"category"`
	Code       string                    `json:"code,omitempty"`
	Message    string                    `json:"message,omitempty"`
	Parameter  string                    `json:"parameter,omitempty"`
	RetryAfter int64                     `json:"retryAfter,omitempty"`
}

func encodeResponseTerminalRecord(record ResponseTerminalRecord) ([]byte, error) {
	if err := validateResponseTerminalRecord(record.Reference, record.Attempt, record.Terminal); err != nil {
		return nil, err
	}
	envelope := responseTerminalEnvelope{
		Schema: responseTerminalEnvelopeSchema, Reference: record.Reference,
		Attempt: responseTerminalAttempt{
			ID: record.Attempt.ID, Number: record.Attempt.Number, BackendID: record.Attempt.BackendID,
			StartedAt: terminalUnixNano(record.Attempt.StartedAt), State: record.Attempt.State,
			StatusCode: record.Attempt.StatusCode, CompletedAt: terminalUnixNano(record.Attempt.CompletedAt),
			ErrorCode: record.Attempt.ErrorCode, FallbackTrigger: record.Attempt.FallbackTrigger,
		},
		Terminal: responseTerminalPayload{
			Usage: record.Terminal.Usage, StopReason: record.Terminal.StopReason,
		},
	}
	if record.Terminal.Error != nil {
		envelope.Terminal.Error = &responseTerminalError{
			Category: record.Terminal.Error.Category, Code: record.Terminal.Error.Code,
			Message: record.Terminal.Error.Message, Parameter: record.Terminal.Error.Parameter,
			RetryAfter: record.Terminal.Error.RetryAfter,
		}
	}
	payload, err := json.Marshal(envelope)
	if err != nil {
		return nil, fmt.Errorf("%w: encode terminal evidence", ErrResponseTerminalInvalid)
	}
	if len(payload) > maximumTerminalPayloadBytes {
		return nil, fmt.Errorf("%w: terminal payload exceeds %d bytes", ErrResponseTerminalInvalid, maximumTerminalPayloadBytes)
	}
	return payload, nil
}

func decodeResponseTerminalRecord(payload []byte) (ResponseTerminalRecord, error) {
	if len(payload) == 0 || len(payload) > maximumTerminalPayloadBytes {
		return ResponseTerminalRecord{}, fmt.Errorf("%w: stored terminal payload size is invalid", ErrResponseTerminalUnavailable)
	}
	var envelope responseTerminalEnvelope
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return ResponseTerminalRecord{}, fmt.Errorf("%w: decode stored terminal evidence", ErrResponseTerminalUnavailable)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return ResponseTerminalRecord{}, fmt.Errorf("%w: stored terminal payload has trailing data", ErrResponseTerminalUnavailable)
	}
	if envelope.Schema != responseTerminalEnvelopeSchema {
		return ResponseTerminalRecord{}, fmt.Errorf("%w: stored terminal envelope is invalid", ErrResponseTerminalUnavailable)
	}
	terminal := ResponseTerminal{
		Usage: envelope.Terminal.Usage, StopReason: envelope.Terminal.StopReason,
	}
	if envelope.Terminal.Error != nil {
		terminal.Error = &llmprotocol.ProtocolError{
			Category: envelope.Terminal.Error.Category, Code: envelope.Terminal.Error.Code,
			Message: envelope.Terminal.Error.Message, Parameter: envelope.Terminal.Error.Parameter,
			RetryAfter: envelope.Terminal.Error.RetryAfter,
		}
	}
	attempt := AttemptResult{
		Attempt: Attempt{
			ID: envelope.Attempt.ID, Number: envelope.Attempt.Number,
			BackendID: envelope.Attempt.BackendID,
			StartedAt: unixNanoTime(envelope.Attempt.StartedAt),
		},
		State: envelope.Attempt.State, StatusCode: envelope.Attempt.StatusCode,
		CompletedAt: unixNanoTime(envelope.Attempt.CompletedAt),
		ErrorCode:   envelope.Attempt.ErrorCode, FallbackTrigger: envelope.Attempt.FallbackTrigger,
	}
	if err := validateResponseTerminalRecord(envelope.Reference, attempt, terminal); err != nil {
		return ResponseTerminalRecord{}, fmt.Errorf("%w: stored terminal record is invalid", ErrResponseTerminalUnavailable)
	}
	return ResponseTerminalRecord{
		Reference: envelope.Reference, Attempt: attempt, Terminal: terminal,
	}, nil
}

func unixNanoTime(value int64) time.Time {
	if value == 0 {
		return time.Time{}
	}
	return time.Unix(0, value).UTC()
}

func terminalUnixNano(value time.Time) int64 {
	if value.IsZero() {
		return 0
	}
	return value.UnixNano()
}
