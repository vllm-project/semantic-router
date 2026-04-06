package milvuslifecycle

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CollectionReader captures collection existence checks.
type CollectionReader interface {
	HasCollection(context.Context, string) (bool, error)
}

// CollectionDropper captures collection deletion.
type CollectionDropper interface {
	DropCollection(context.Context, string, ...client.DropCollectionOption) error
}

// CollectionCreator captures collection creation.
type CollectionCreator interface {
	CreateCollection(context.Context, *entity.Schema, int32, ...client.CreateCollectionOption) error
}

// CollectionLoader captures collection loading.
type CollectionLoader interface {
	LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error
}

// IndexCreator captures index creation.
type IndexCreator interface {
	CreateIndex(context.Context, string, string, entity.Index, bool, ...client.IndexOption) error
}

// IndexDescriber captures index inspection for existing collections.
type IndexDescriber interface {
	DescribeIndex(context.Context, string, string, ...client.IndexOption) ([]entity.Index, error)
}

// BootstrapClient captures the create, index, and load operations needed for bootstrap.
type BootstrapClient interface {
	CollectionCreator
	CollectionLoader
	IndexCreator
}

// ExistingIndexClient captures the index operations needed when reconciling existing collections.
type ExistingIndexClient interface {
	IndexCreator
	IndexDescriber
}

// EnsureClient captures the full shared collection lifecycle surface.
type EnsureClient interface {
	CollectionReader
	CollectionDropper
	BootstrapClient
	ExistingIndexClient
}

// IndexSpec describes one Milvus index owned by a collection lifecycle spec.
type IndexSpec struct {
	FieldName        string
	Build            func() (entity.Index, error)
	EnsureOnExisting bool
}

// CollectionSpec describes the shared lifecycle work needed to bootstrap a collection.
type CollectionSpec struct {
	Name                      string
	Schema                    *entity.Schema
	ShardNum                  int32
	CreateOptions             []client.CreateCollectionOption
	Indexes                   []IndexSpec
	Load                      bool
	IgnoreLoadErrorOnExisting bool
}

// EnsureOptions controls how EnsureCollection reconciles existing state.
type EnsureOptions struct {
	AllowCreate  bool
	DropExisting bool
}

// EnsureResult reports what lifecycle actions were taken.
type EnsureResult struct {
	Created bool
	Existed bool
	Dropped bool
}

// RetryPolicy defines a small shared retry loop for Milvus lifecycle operations.
type RetryPolicy struct {
	Attempts    int
	Backoff     func(attempt int) time.Duration
	ShouldRetry func(error) bool
}

// CollectionExists checks whether a collection already exists.
func CollectionExists(ctx context.Context, lifecycleClient CollectionReader, collectionName string) (bool, error) {
	exists, err := lifecycleClient.HasCollection(ctx, collectionName)
	if err != nil {
		return false, fmt.Errorf("failed to check collection existence: %w", err)
	}
	return exists, nil
}

// CreateCollection creates indexes and loads a collection using one shared lifecycle path.
func CreateCollection(ctx context.Context, lifecycleClient BootstrapClient, spec CollectionSpec) error {
	collectionName, err := validateCollectionSpec(spec)
	if err != nil {
		return err
	}

	shardNum := spec.ShardNum
	if shardNum <= 0 {
		shardNum = 1
	}

	if err := lifecycleClient.CreateCollection(ctx, spec.Schema, shardNum, spec.CreateOptions...); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	if err := createIndexes(ctx, lifecycleClient, collectionName, spec.Indexes); err != nil {
		return err
	}

	if spec.Load {
		if err := lifecycleClient.LoadCollection(ctx, collectionName, false); err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}
	}

	return nil
}

// EnsureCollection reconciles one collection against a shared lifecycle policy.
func EnsureCollection(
	ctx context.Context, lifecycleClient EnsureClient, spec CollectionSpec, options EnsureOptions,
) (EnsureResult, error) {
	var result EnsureResult

	collectionName, err := validateCollectionSpec(spec)
	if err != nil {
		return result, err
	}

	exists, err := CollectionExists(ctx, lifecycleClient, collectionName)
	if err != nil {
		return result, err
	}

	if exists {
		result.Existed = true
	}

	if options.DropExisting && exists {
		if err := lifecycleClient.DropCollection(ctx, collectionName); err != nil {
			return result, fmt.Errorf("failed to drop collection: %w", err)
		}
		result.Dropped = true
		result.Existed = false
		exists = false
	}

	if !exists {
		if !options.AllowCreate {
			return result, fmt.Errorf("collection %s does not exist and auto-creation is disabled", collectionName)
		}
		if err := CreateCollection(ctx, lifecycleClient, spec); err != nil {
			return result, err
		}
		result.Created = true
		return result, nil
	}

	if err := ensureExistingIndexes(ctx, lifecycleClient, collectionName, spec.Indexes); err != nil {
		return result, err
	}

	if spec.Load {
		if err := lifecycleClient.LoadCollection(ctx, collectionName, false); err != nil {
			if spec.IgnoreLoadErrorOnExisting {
				return result, nil
			}
			return result, fmt.Errorf("failed to load collection: %w", err)
		}
	}

	return result, nil
}

// Retry runs a small reusable retry loop with context cancellation support.
func Retry(ctx context.Context, policy RetryPolicy, operation func() error) error {
	attempts := policy.Attempts
	if attempts <= 0 {
		attempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		lastErr = operation()
		if lastErr == nil {
			return nil
		}
		if policy.ShouldRetry != nil && !policy.ShouldRetry(lastErr) {
			return lastErr
		}
		if attempt == attempts-1 {
			return lastErr
		}

		delay := time.Duration(0)
		if policy.Backoff != nil {
			delay = policy.Backoff(attempt)
		}
		if delay <= 0 {
			continue
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
		}
	}

	return lastErr
}

func validateCollectionSpec(spec CollectionSpec) (string, error) {
	if spec.Schema == nil {
		return "", fmt.Errorf("milvus collection schema is required")
	}

	collectionName := spec.Name
	if collectionName == "" {
		collectionName = spec.Schema.CollectionName
	}
	if collectionName == "" {
		return "", fmt.Errorf("milvus collection name is required")
	}
	if spec.Schema.CollectionName == "" {
		spec.Schema.CollectionName = collectionName
	}

	return collectionName, nil
}

func createIndexes(
	ctx context.Context, lifecycleClient IndexCreator, collectionName string, indexes []IndexSpec,
) error {
	for _, spec := range indexes {
		if err := createIndex(ctx, lifecycleClient, collectionName, spec); err != nil {
			return err
		}
	}
	return nil
}

func ensureExistingIndexes(
	ctx context.Context, lifecycleClient ExistingIndexClient, collectionName string, indexes []IndexSpec,
) error {
	for _, spec := range indexes {
		if !spec.EnsureOnExisting {
			continue
		}

		existing, err := lifecycleClient.DescribeIndex(ctx, collectionName, spec.FieldName)
		if err != nil {
			return fmt.Errorf("failed to describe index on %s: %w", spec.FieldName, err)
		}
		if len(existing) > 0 {
			continue
		}

		if err := createIndex(ctx, lifecycleClient, collectionName, spec); err != nil {
			return err
		}
	}
	return nil
}

func createIndex(ctx context.Context, lifecycleClient IndexCreator, collectionName string, spec IndexSpec) error {
	if spec.FieldName == "" {
		return fmt.Errorf("milvus index field name is required")
	}
	if spec.Build == nil {
		return fmt.Errorf("milvus index builder is required for field %s", spec.FieldName)
	}

	index, err := spec.Build()
	if err != nil {
		return fmt.Errorf("failed to build index on %s: %w", spec.FieldName, err)
	}
	if err := lifecycleClient.CreateIndex(ctx, collectionName, spec.FieldName, index, false); err != nil {
		return fmt.Errorf("failed to create index on %s: %w", spec.FieldName, err)
	}

	return nil
}
