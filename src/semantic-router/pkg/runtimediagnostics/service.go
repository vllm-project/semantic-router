// Package runtimediagnostics composes sanitized health observations from the
// Router's authoritative PostgreSQL and applied Valkey state. It never exposes
// connection strings, runtime keys, credentials, or policy documents.
package runtimediagnostics

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

var ErrNotFound = errors.New("runtime diagnostics namespace not found")

const usageStorageDiagnosticBucketLimit int64 = 10000

type PublicationReader interface {
	CountPublicationNamespaces(context.Context) (int64, error)
	GetPublicationNamespace(context.Context, string) (accesspublisher.NamespacePublication, error)
	Diagnostics(context.Context, string, string) (accesspublisher.RuntimeDiagnostics, error)
}

type QuotaReader interface {
	Snapshot(context.Context, string) (quotaruntime.PartitionDiagnostics, error)
}

type StoreStatus struct {
	Status string `json:"status"`
}

type UsageStorageStatus struct {
	Status             string     `json:"status"`
	ActiveMonths       int64      `json:"activeMonths"`
	RetiredMonths      int64      `json:"retiredMonths"`
	DirtyMinuteBuckets int64      `json:"dirtyMinuteBuckets"`
	DirtyHourBuckets   int64      `json:"dirtyHourBuckets"`
	DirtyDayBuckets    int64      `json:"dirtyDayBuckets"`
	DirtyCountsCapped  bool       `json:"dirtyCountsCapped"`
	OldestActiveMonth  *time.Time `json:"oldestActiveMonth,omitempty"`
	CreatedThrough     *time.Time `json:"createdThrough,omitempty"`
}

type NamespaceDiagnostics struct {
	NamespaceID                    string                             `json:"namespaceId"`
	QuotaPartition                 string                             `json:"quotaPartition"`
	Publication                    accesspublisher.RuntimeDiagnostics `json:"publication"`
	Quota                          quotaruntime.PartitionDiagnostics  `json:"quota"`
	UsageStreamBacklogLimit        int64                              `json:"usageStreamBacklogLimit"`
	AdmissionBlockedByUsageBacklog bool                               `json:"admissionBlockedByUsageBacklog"`
}

type Snapshot struct {
	Status               string                `json:"status"`
	AsOf                 time.Time             `json:"asOf"`
	PostgreSQL           StoreStatus           `json:"postgresql"`
	Valkey               StoreStatus           `json:"valkey"`
	UsageStorage         UsageStorageStatus    `json:"usageStorage"`
	RegisteredNamespaces int64                 `json:"registeredNamespaces"`
	Namespace            *NamespaceDiagnostics `json:"namespace,omitempty"`
}

type Options struct {
	Database        *sql.DB
	Valkey          *redis.Client
	Publications    PublicationReader
	Quota           QuotaReader
	MaxUsageBacklog int64
	Now             func() time.Time
}

type Service struct {
	database        *sql.DB
	valkey          *redis.Client
	publications    PublicationReader
	quota           QuotaReader
	maxUsageBacklog int64
	now             func() time.Time
}

func New(options Options) (*Service, error) {
	if options.Database == nil || options.Valkey == nil || options.Publications == nil || options.Quota == nil || options.MaxUsageBacklog < 1 {
		return nil, errors.New("runtime diagnostics dependencies are incomplete")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Service{
		database: options.Database, valkey: options.Valkey, publications: options.Publications,
		quota: options.Quota, maxUsageBacklog: options.MaxUsageBacklog, now: now,
	}, nil
}

// Read returns partial, sanitized health state even when a store is down. A
// namespace filter is an exact diagnostic selector; absence never falls back
// to another namespace or scans policy rows.
func (service *Service) Read(ctx context.Context, namespaceID string) (Snapshot, error) {
	if service == nil || service.database == nil || service.valkey == nil {
		return Snapshot{}, errors.New("runtime diagnostics are unavailable")
	}
	result := Snapshot{
		Status: "ready", AsOf: service.now().UTC(),
		PostgreSQL: StoreStatus{Status: "ready"}, Valkey: StoreStatus{Status: "ready"},
		UsageStorage: UsageStorageStatus{Status: "ready"},
	}
	if err := service.database.PingContext(ctx); err != nil {
		result.PostgreSQL.Status = "unavailable"
		result.UsageStorage.Status = "unavailable"
		result.Status = "degraded"
	} else if storage, err := service.readUsageStorage(ctx); err != nil {
		result.UsageStorage.Status = "unavailable"
		result.Status = "degraded"
	} else {
		result.UsageStorage = storage
	}
	if err := service.valkey.Ping(ctx).Err(); err != nil {
		result.Valkey.Status = "unavailable"
		result.Status = "degraded"
		return result, nil
	}
	namespaceCount, err := service.publications.CountPublicationNamespaces(ctx)
	if err != nil {
		result.Valkey.Status = "unavailable"
		result.Status = "degraded"
		return result, nil
	}
	result.RegisteredNamespaces = namespaceCount
	if namespaceID == "" {
		return result, nil
	}
	selected, err := service.publications.GetPublicationNamespace(ctx, namespaceID)
	if errors.Is(err, accesspublisher.ErrNamespaceNotFound) {
		return Snapshot{}, ErrNotFound
	}
	if err != nil {
		return Snapshot{}, fmt.Errorf("read publication namespace: %w", err)
	}
	publication, err := service.publications.Diagnostics(ctx, selected.NamespaceID, selected.QuotaPartition)
	if err != nil {
		return Snapshot{}, fmt.Errorf("read publication diagnostics: %w", err)
	}
	quotaState, err := service.quota.Snapshot(ctx, selected.QuotaPartition)
	if err != nil {
		return Snapshot{}, fmt.Errorf("read quota diagnostics: %w", err)
	}
	result.AsOf = publication.AsOf
	if quotaState.AsOf.After(result.AsOf) {
		result.AsOf = quotaState.AsOf
	}
	result.Namespace = &NamespaceDiagnostics{
		NamespaceID: selected.NamespaceID, QuotaPartition: selected.QuotaPartition,
		Publication: publication, Quota: quotaState,
		UsageStreamBacklogLimit:        service.maxUsageBacklog,
		AdmissionBlockedByUsageBacklog: quotaState.UsageStreamBacklog >= service.maxUsageBacklog,
	}
	if !publication.Readiness.Ready || quotaState.RecoveryState != "ready" || result.Namespace.AdmissionBlockedByUsageBacklog {
		result.Status = "degraded"
	}
	return result, nil
}

func (service *Service) readUsageStorage(ctx context.Context) (UsageStorageStatus, error) {
	var result UsageStorageStatus
	var oldest, createdThrough sql.NullTime
	err := service.database.QueryRowContext(ctx, `SELECT
  count(*) FILTER (WHERE state='active'),
  count(*) FILTER (WHERE state='retired'),
  min(month_start) FILTER (WHERE state='active'),
  max(month_start) FILTER (WHERE state='active'),
	(SELECT count(*) FROM (
	  SELECT 1 FROM usage_rollup_dirty_minutes LIMIT $1
	) bounded_dirty_minutes),
	(SELECT count(*) FROM (
	  SELECT 1 FROM usage_rollup_dirty_hours LIMIT $1
	) bounded_dirty_hours),
	(SELECT count(*) FROM (
	  SELECT 1 FROM usage_rollup_dirty_days LIMIT $1
	) bounded_dirty_days)
FROM usage_partition_months`, usageStorageDiagnosticBucketLimit+1).Scan(
		&result.ActiveMonths, &result.RetiredMonths, &oldest, &createdThrough,
		&result.DirtyMinuteBuckets, &result.DirtyHourBuckets, &result.DirtyDayBuckets,
	)
	if err != nil {
		return UsageStorageStatus{}, fmt.Errorf("read usage storage diagnostics: %w", err)
	}
	result.Status = "ready"
	for _, count := range []*int64{
		&result.DirtyMinuteBuckets, &result.DirtyHourBuckets, &result.DirtyDayBuckets,
	} {
		if *count > usageStorageDiagnosticBucketLimit {
			*count = usageStorageDiagnosticBucketLimit
			result.DirtyCountsCapped = true
		}
	}
	if oldest.Valid {
		value := oldest.Time.UTC()
		result.OldestActiveMonth = &value
	}
	if createdThrough.Valid {
		value := createdThrough.Time.UTC()
		result.CreatedThrough = &value
	}
	return result, nil
}
