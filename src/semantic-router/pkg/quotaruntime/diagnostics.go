package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

// PartitionDiagnostics is a read-only, point-in-time observation of the
// durable quota work queues. It never drains pending work or acknowledges a
// usage-stream item.
type PartitionDiagnostics struct {
	Partition                string     `json:"partition"`
	AsOf                     time.Time  `json:"asOf"`
	UsageStreamBacklog       int64      `json:"usageStreamBacklog"`
	PendingAdmissions        int64      `json:"pendingAdmissions"`
	ExpiredPendingAdmissions int64      `json:"expiredPendingAdmissions"`
	OldestPendingDeadline    *time.Time `json:"oldestPendingDeadline,omitempty"`
	RecoveryState            string     `json:"recoveryState"`
}

// RedisDiagnostics reads the same partition-local keys used by admission. It
// deliberately exposes typed observations rather than raw Redis key names.
type RedisDiagnostics struct {
	client    redis.UniversalClient
	keyPrefix string
}

func NewRedisDiagnostics(client redis.UniversalClient, keyPrefix string) (*RedisDiagnostics, error) {
	if client == nil {
		return nil, fmt.Errorf("%w: Redis diagnostics client is required", ErrInvalidRequest)
	}
	if _, err := newPartitionKeysWithPrefix(keyPrefix, "validation"); err != nil {
		return nil, err
	}
	return &RedisDiagnostics{client: client, keyPrefix: keyPrefix}, nil
}

func (diagnostics *RedisDiagnostics) Snapshot(
	ctx context.Context,
	partition string,
) (PartitionDiagnostics, error) {
	if diagnostics == nil || diagnostics.client == nil {
		return PartitionDiagnostics{}, fmt.Errorf("%w: quota diagnostics are unavailable", ErrRuntimeUnavailable)
	}
	keys, err := newPartitionKeysWithPrefix(diagnostics.keyPrefix, partition)
	if err != nil {
		return PartitionDiagnostics{}, err
	}
	asOf, err := diagnostics.client.Time(ctx).Result()
	if err != nil {
		return PartitionDiagnostics{}, fmt.Errorf("%w: read quota runtime time", ErrRuntimeUnavailable)
	}
	usageBacklog, err := consumerGroupBacklog(
		ctx, diagnostics.client, keys.usageStream, usageledger.ConsumerGroupName,
	)
	if err != nil {
		return PartitionDiagnostics{}, fmt.Errorf("%w: read usage accounting backlog: %v", ErrRuntimeUnavailable, err)
	}
	pipeline := diagnostics.client.Pipeline()
	pending := pipeline.ZCard(ctx, keys.pendingIndex)
	expired := pipeline.ZCount(ctx, keys.pendingIndex, "-inf", strconv.FormatInt(asOf.UnixMilli(), 10))
	oldest := pipeline.ZRangeWithScores(ctx, keys.pendingIndex, 0, 0)
	_, err = pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return PartitionDiagnostics{}, fmt.Errorf("%w: read quota runtime queues", ErrRuntimeUnavailable)
	}
	result := PartitionDiagnostics{
		Partition: partition, AsOf: asOf.UTC(), UsageStreamBacklog: usageBacklog,
		PendingAdmissions: pending.Val(), ExpiredPendingAdmissions: expired.Val(),
		RecoveryState: "ready",
	}
	if values := oldest.Val(); len(values) > 0 {
		deadline := time.UnixMilli(int64(values[0].Score)).UTC()
		result.OldestPendingDeadline = &deadline
	}
	if result.ExpiredPendingAdmissions > 0 {
		result.RecoveryState = "reconciliation_required"
	}
	return result, nil
}

func consumerGroupBacklog(
	ctx context.Context,
	client redis.UniversalClient,
	stream string,
	group string,
) (int64, error) {
	streamType, err := client.Type(ctx, stream).Result()
	if err != nil {
		return 0, err
	}
	if streamType == "none" {
		return 0, nil
	}
	if streamType != "stream" {
		return 0, fmt.Errorf("usage stream key has type %q", streamType)
	}
	groups, err := client.XInfoGroups(ctx, stream).Result()
	if err != nil {
		return 0, err
	}
	for _, candidate := range groups {
		if candidate.Name != group {
			continue
		}
		if candidate.Pending < 0 || candidate.Lag < 0 {
			return 0, fmt.Errorf("usage consumer group backlog is indeterminate")
		}
		return candidate.Pending + candidate.Lag, nil
	}
	return 0, fmt.Errorf("usage consumer group %q is unavailable", group)
}
