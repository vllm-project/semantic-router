package managedruntime

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const publicationProjectorID = "router-publication"

func composePublicationPipeline(
	database *sql.DB,
	client *redis.Client,
	store config.RedisAccessRuntimeStoreConfig,
	replicaID string,
) (accesspublisher.Processor, *accesspublisher.Worker, *accesspublisher.RedisStore, error) {
	outbox, err := accesspublisher.NewPostgresStore(database, accesspublisher.PostgresStoreOptions{
		Projector: publicationProjectorID,
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("compose publication outbox: %w", err)
	}
	desired, err := accesspublisher.NewPostgresDesiredStateReader(database)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("compose publication desired-state reader: %w", err)
	}
	applied, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: client, KeyPrefix: store.KeyPrefix, RequireFleetReplicas: true,
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("compose publication runtime store: %w", err)
	}
	engine, err := accesspublisher.NewEngine(accesspublisher.EngineOptions{
		Outbox: outbox, Desired: desired, Runtime: applied, WorkerID: replicaID,
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("compose publication engine: %w", err)
	}
	worker, err := accesspublisher.NewWorker(accesspublisher.WorkerOptions{Processor: engine})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("compose publication worker: %w", err)
	}
	return engine, worker, applied, nil
}

func composeUsageSupervisor(
	database *sql.DB,
	client *redis.Client,
	store config.RedisAccessRuntimeStoreConfig,
	storageConfig config.AccessUsageStorageConfig,
	replicaID string,
) (*usageledger.Supervisor, error) {
	maintenanceInterval, err := time.ParseDuration(storageConfig.MaintenanceInterval)
	if err != nil {
		return nil, fmt.Errorf("parse usage maintenance interval: %w", err)
	}
	var rawRetention time.Duration
	if storageConfig.RawRetention != "" {
		rawRetention, err = time.ParseDuration(storageConfig.RawRetention)
		if err != nil {
			return nil, fmt.Errorf("parse usage raw retention: %w", err)
		}
	}
	storage, err := usageledger.NewPostgresStorageLifecycle(database, usageledger.StorageLifecycleOptions{
		CreateAheadMonths:   storageConfig.CreateAheadMonths,
		MaintenanceInterval: maintenanceInterval,
		RawRetention:        rawRetention,
	})
	if err != nil {
		return nil, err
	}
	namespaces, err := usageledger.NewPostgresNamespaceSource(database)
	if err != nil {
		return nil, err
	}
	streams, err := usageledger.NewRedisStreamFactory(client, store.KeyPrefix)
	if err != nil {
		return nil, err
	}
	rollups, err := usageledger.NewPostgresRollupProcessor(
		database,
		usageledger.PostgresRollupProcessorOptions{},
	)
	if err != nil {
		return nil, err
	}
	supervisor, err := usageledger.NewSupervisor(usageledger.SupervisorOptions{
		Namespaces: namespaces,
		Streams:    streams,
		Store:      usageledger.PostgresStore{DB: database, Partitions: storage},
		Rollups:    rollups,
		Storage:    storage,
		ReplicaID:  replicaID,
	})
	if err != nil {
		return nil, fmt.Errorf("compose usage supervisor: %w", err)
	}
	return supervisor, nil
}
