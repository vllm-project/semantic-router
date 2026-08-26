package usageledger

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

var (
	partitionPattern        = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	keyPrefixPattern        = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*(:[A-Za-z0-9][A-Za-z0-9._-]*)*$`)
	quarantineReasonPattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
)

const (
	// ConsumerGroupName is the canonical group shared by usage writers and
	// admission backpressure. Keeping one name makes lag plus pending the
	// authoritative distributed backlog across every Router replica.
	ConsumerGroupName         = "usage-writers"
	maxQuarantinePayloadBytes = maxEventBytes + 4096
)

type StreamItem struct {
	ID     string
	Values map[string]string
}

type Stream interface {
	EnsureGroup(context.Context) error
	ReadNew(context.Context, int64, time.Duration) ([]StreamItem, error)
	ClaimStale(context.Context, int64, time.Duration) ([]StreamItem, error)
	Ack(context.Context, []string) error
	Quarantine(context.Context, StreamItem, string) (bool, error)
	Quarantined(context.Context) (int64, error)
}

type RedisStreamOptions struct {
	KeyPrefix string
	Partition string
	Group     string
	Consumer  string
}

type RedisStream struct {
	client        redis.UniversalClient
	key           string
	quarantineKey string
	group         string
	consumer      string
}

func NewRedisStream(client redis.UniversalClient, options RedisStreamOptions) (*RedisStream, error) {
	if client == nil {
		return nil, fmt.Errorf("usage stream Redis client is required")
	}
	if options.KeyPrefix != "" && (len(options.KeyPrefix) > 128 || !keyPrefixPattern.MatchString(options.KeyPrefix)) {
		return nil, fmt.Errorf("usage stream key prefix is not canonical")
	}
	if !partitionPattern.MatchString(options.Partition) {
		return nil, fmt.Errorf("usage stream partition is not canonical")
	}
	if err := validateConsumerName("group", options.Group); err != nil {
		return nil, err
	}
	if err := validateConsumerName("consumer", options.Consumer); err != nil {
		return nil, err
	}
	key := "usage-stream:{" + options.Partition + "}"
	quarantineKey := "usage-quarantine:{" + options.Partition + "}"
	if options.KeyPrefix != "" {
		key = options.KeyPrefix + ":" + key
		quarantineKey = options.KeyPrefix + ":" + quarantineKey
	}
	return &RedisStream{
		client: client, key: key, quarantineKey: quarantineKey,
		group: options.Group, consumer: options.Consumer,
	}, nil
}

func (s *RedisStream) EnsureGroup(ctx context.Context) error {
	err := s.client.XGroupCreateMkStream(ctx, s.key, s.group, "0").Err()
	if err != nil && !strings.Contains(err.Error(), "BUSYGROUP") {
		return fmt.Errorf("create usage consumer group: %w", err)
	}
	if err := s.client.XGroupCreateConsumer(ctx, s.key, s.group, s.consumer).Err(); err != nil {
		return fmt.Errorf("create usage stream consumer: %w", err)
	}
	return nil
}

func (s *RedisStream) ReadNew(ctx context.Context, count int64, block time.Duration) ([]StreamItem, error) {
	if count <= 0 || count > 1000 {
		return nil, fmt.Errorf("usage stream read count must be between 1 and 1000")
	}
	if block < 0 || block > time.Minute {
		return nil, fmt.Errorf("usage stream block duration must be between zero and one minute")
	}
	streams, err := s.client.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group: s.group, Consumer: s.consumer, Streams: []string{s.key, ">"}, Count: count, Block: block,
	}).Result()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read usage stream: %w", err)
	}
	if len(streams) != 1 || streams[0].Stream != s.key {
		return nil, fmt.Errorf("usage stream returned an unexpected stream")
	}
	return convertMessages(streams[0].Messages)
}

func (s *RedisStream) ClaimStale(ctx context.Context, count int64, minIdle time.Duration) ([]StreamItem, error) {
	if count <= 0 || count > 1000 {
		return nil, fmt.Errorf("usage stream reclaim count must be between 1 and 1000")
	}
	if minIdle <= 0 {
		return nil, fmt.Errorf("usage stream reclaim idle time must be positive")
	}
	messages, _, err := s.client.XAutoClaim(ctx, &redis.XAutoClaimArgs{
		Stream: s.key, Group: s.group, Consumer: s.consumer, MinIdle: minIdle, Start: "0-0", Count: count,
	}).Result()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reclaim usage stream: %w", err)
	}
	return convertMessages(messages)
}

func (s *RedisStream) Ack(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	if len(ids) > 1000 {
		return fmt.Errorf("usage stream acknowledgement exceeds 1000 items")
	}
	for _, id := range ids {
		if err := validateConsumerName("stream ID", id); err != nil {
			return err
		}
	}
	args := make([]any, 0, len(ids)+1)
	args = append(args, s.group)
	for _, id := range ids {
		args = append(args, id)
	}
	count, err := acknowledgeUsageScript.Run(ctx, s.client, []string{s.key}, args...).Int64()
	if err != nil {
		return fmt.Errorf("acknowledge usage stream: %w", err)
	}
	if count != int64(len(ids)) {
		return fmt.Errorf("acknowledge usage stream: acknowledged %d of %d items", count, len(ids))
	}
	return nil
}

// Quarantine durably moves one pending malformed item out of the admission
// backlog. The bounded original payload is retained before the source item is
// acknowledged and deleted, so operators can recover accounting evidence
// without one poisoned record preventing later valid records from ingesting.
func (s *RedisStream) Quarantine(
	ctx context.Context,
	item StreamItem,
	reason string,
) (bool, error) {
	if err := validateConsumerName("stream ID", item.ID); err != nil {
		return false, err
	}
	if !quarantineReasonPattern.MatchString(reason) {
		return false, fmt.Errorf("usage quarantine reason is not canonical")
	}
	payload, err := json.Marshal(item.Values)
	if err != nil {
		return false, fmt.Errorf("encode usage quarantine payload: %w", err)
	}
	if len(payload) == 0 || len(payload) > maxQuarantinePayloadBytes {
		return false, fmt.Errorf("usage quarantine payload exceeds its bounded envelope")
	}
	digest := sha256.Sum256(payload)
	value, err := quarantineUsageScript.Run(ctx, s.client, []string{s.key, s.quarantineKey},
		s.group, item.ID, reason, string(payload), hex.EncodeToString(digest[:]),
	).Result()
	if err != nil {
		return false, fmt.Errorf("quarantine usage stream item: %w", err)
	}
	fields, ok := value.([]any)
	if !ok || len(fields) != 2 {
		return false, fmt.Errorf("quarantine usage stream item returned an invalid result")
	}
	moved, ok := fields[0].(int64)
	if !ok || (moved != 0 && moved != 1) {
		return false, fmt.Errorf("quarantine usage stream item returned an invalid state")
	}
	return moved == 1, nil
}

// Quarantined reports the durable number of malformed items retained for
// operator inspection. Quarantined payloads never re-enter normal ingestion.
func (s *RedisStream) Quarantined(ctx context.Context) (int64, error) {
	count, err := s.client.XLen(ctx, s.quarantineKey).Result()
	if err != nil {
		return 0, fmt.Errorf("read usage quarantine backlog: %w", err)
	}
	return count, nil
}

func convertMessages(messages []redis.XMessage) ([]StreamItem, error) {
	result := make([]StreamItem, 0, len(messages))
	for _, message := range messages {
		values := make(map[string]string, len(message.Values))
		for key, raw := range message.Values {
			switch value := raw.(type) {
			case string:
				values[key] = value
			case []byte:
				values[key] = string(value)
			default:
				return nil, fmt.Errorf("usage stream item %q field %q is not a string", message.ID, key)
			}
		}
		result = append(result, StreamItem{ID: message.ID, Values: values})
	}
	return result, nil
}

func validateConsumerName(label, value string) error {
	if value == "" || len(value) > 128 || strings.TrimSpace(value) != value || strings.ContainsAny(value, "\x00 \t\r\n") {
		return fmt.Errorf("usage stream %s is not a bounded canonical identifier", label)
	}
	return nil
}
