package accesscapacity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type commandSnapshot map[string]int64

func readCommandSnapshot(ctx context.Context, client *redis.Client) (commandSnapshot, error) {
	payload, err := client.Info(ctx, "commandstats").Result()
	if err != nil {
		return nil, fmt.Errorf("read Redis command statistics: %w", err)
	}
	result := make(commandSnapshot)
	for _, line := range strings.Split(payload, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "cmdstat_") {
			continue
		}
		name, fields, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		for _, field := range strings.Split(fields, ",") {
			value, ok := strings.CutPrefix(field, "calls=")
			if !ok {
				continue
			}
			calls, parseErr := strconv.ParseInt(value, 10, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("parse Redis command statistics for %s: %w", name, parseErr)
			}
			result[strings.TrimPrefix(name, "cmdstat_")] = calls
		}
	}
	return result, nil
}

func commandDelta(before, after commandSnapshot) RedisOperation {
	result := RedisOperation{ByCommand: make(map[string]int64)}
	for command, value := range after {
		delta := value - before[command]
		if delta <= 0 {
			continue
		}
		result.ByCommand[command] = delta
		result.Total += delta
	}
	return result
}

type memorySnapshot struct {
	Keys  int64
	Bytes int64
}

func readMemorySnapshot(ctx context.Context, client *redis.Client, prefix string) (memorySnapshot, error) {
	var result memorySnapshot
	var cursor uint64
	pattern := prefix + ":*"
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, 1000).Result()
		if err != nil {
			return memorySnapshot{}, fmt.Errorf("scan capacity keyspace: %w", err)
		}
		for start := 0; start < len(keys); start += 256 {
			end := min(start+256, len(keys))
			pipeline := client.Pipeline()
			commands := make([]*redis.IntCmd, 0, end-start)
			for _, key := range keys[start:end] {
				commands = append(commands, pipeline.MemoryUsage(ctx, key, 0))
			}
			if _, err := pipeline.Exec(ctx); err != nil {
				return memorySnapshot{}, fmt.Errorf("measure capacity keyspace: %w", err)
			}
			for _, command := range commands {
				result.Bytes += command.Val()
				result.Keys++
			}
		}
		cursor = next
		if cursor == 0 {
			break
		}
	}
	return result, nil
}

func deletePrefix(ctx context.Context, client *redis.Client, prefix string) error {
	var cursor uint64
	pattern := prefix + ":*"
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, 1000).Result()
		if err != nil {
			return fmt.Errorf("scan capacity keyspace for cleanup: %w", err)
		}
		for start := 0; start < len(keys); start += 1000 {
			end := min(start+1000, len(keys))
			if err := client.Unlink(ctx, keys[start:end]...).Err(); err != nil {
				return fmt.Errorf("remove capacity keyspace: %w", err)
			}
		}
		cursor = next
		if cursor == 0 {
			return nil
		}
	}
}

func readRedisEnvironment(ctx context.Context, client *redis.Client) (string, string, error) {
	payload, err := client.Info(ctx, "server").Result()
	if err != nil {
		return "", "", fmt.Errorf("read Redis server information: %w", err)
	}
	version, mode := "", ""
	for _, line := range strings.Split(payload, "\n") {
		line = strings.TrimSpace(line)
		if value, found := strings.CutPrefix(line, "redis_version:"); found {
			version = value
		}
		if value, found := strings.CutPrefix(line, "redis_mode:"); found {
			mode = value
		}
	}
	return version, mode, nil
}

type usageEvent struct {
	EmittedAtUnixNano int64 `json:"emittedAtUnixNano"`
}

type usageObservation struct {
	Observed     int64
	Acknowledged int64
	Lag          []time.Duration
	Err          error
}

func startUsageObserver(
	ctx context.Context,
	client *redis.Client,
	prefix string,
	expected int64,
) (<-chan usageObservation, error) {
	stream, err := usageledger.NewRedisStream(client, usageledger.RedisStreamOptions{
		KeyPrefix: prefix, Partition: fixturePartition,
		Group: "capacity-gate", Consumer: "capacity-observer",
	})
	if err != nil {
		return nil, err
	}
	if err := stream.EnsureGroup(ctx); err != nil {
		return nil, err
	}
	result := make(chan usageObservation, 1)
	go func() {
		observation := usageObservation{Lag: make([]time.Duration, 0, expected)}
		defer func() { result <- observation }()
		for observation.Observed < expected {
			items, readErr := stream.ReadNew(ctx, 1000, 250*time.Millisecond)
			if readErr != nil {
				if errors.Is(readErr, context.Canceled) || errors.Is(readErr, context.DeadlineExceeded) {
					observation.Err = ctx.Err()
				} else {
					observation.Err = readErr
				}
				return
			}
			if len(items) == 0 {
				continue
			}
			ids := make([]string, 0, len(items))
			observedAt := time.Now()
			for _, item := range items {
				var event usageEvent
				if decodeErr := json.Unmarshal([]byte(item.Values["event"]), &event); decodeErr != nil || event.EmittedAtUnixNano <= 0 {
					observation.Err = fmt.Errorf("decode usage event %s", item.ID)
					return
				}
				emittedAt := time.Unix(0, event.EmittedAtUnixNano)
				lag := observedAt.Sub(emittedAt)
				if lag < 0 {
					lag = 0
				}
				observation.Lag = append(observation.Lag, lag)
				ids = append(ids, item.ID)
				observation.Observed++
			}
			if ackErr := stream.Ack(ctx, ids); ackErr != nil {
				observation.Err = ackErr
				return
			}
			observation.Acknowledged += int64(len(ids))
		}
	}()
	return result, nil
}

func usageGroupState(ctx context.Context, client *redis.Client, prefix string) (retained, pending, lag int64, err error) {
	streamKey := prefix + ":usage-stream:{" + fixturePartition + "}"
	retained, err = client.XLen(ctx, streamKey).Result()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read usage stream length: %w", err)
	}
	groups, err := client.XInfoGroups(ctx, streamKey).Result()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read usage stream groups: %w", err)
	}
	for _, group := range groups {
		if group.Name == "capacity-gate" {
			return retained, group.Pending, max(group.Lag, 0), nil
		}
	}
	return 0, 0, 0, fmt.Errorf("capacity usage consumer group is absent")
}
