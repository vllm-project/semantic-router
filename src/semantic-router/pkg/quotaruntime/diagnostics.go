package quotaruntime

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
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
	pipeline := diagnostics.client.Pipeline()
	usageBacklog := pipeline.XLen(ctx, keys.usageStream)
	pending := pipeline.ZCard(ctx, keys.pendingIndex)
	expired := pipeline.ZCount(ctx, keys.pendingIndex, "-inf", strconv.FormatInt(asOf.UnixMilli(), 10))
	oldest := pipeline.ZRangeWithScores(ctx, keys.pendingIndex, 0, 0)
	_, err = pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return PartitionDiagnostics{}, fmt.Errorf("%w: read quota runtime queues", ErrRuntimeUnavailable)
	}
	result := PartitionDiagnostics{
		Partition: partition, AsOf: asOf.UTC(), UsageStreamBacklog: usageBacklog.Val(),
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
