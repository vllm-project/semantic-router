package milvus

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// LifecycleClient captures the subset of Milvus client lifecycle APIs reused by
// runtime stores.
type LifecycleClient interface {
	HasCollection(ctx context.Context, collectionName string) (bool, error)
	LoadCollection(ctx context.Context, collectionName string, async bool, opts ...client.LoadCollectionOption) error
}

// ConnectGRPC creates a Milvus gRPC client with an optional timeout.
func ConnectGRPC(ctx context.Context, address string, timeout time.Duration) (client.Client, error) {
	if address == "" {
		return nil, fmt.Errorf("milvus address is required")
	}

	dialCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		dialCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	milvusClient, err := client.NewGrpcClient(dialCtx, address)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Milvus at %s: %w", address, err)
	}

	return milvusClient, nil
}

// Connect creates a Milvus SDK client (with username/password support) with an optional timeout.
func Connect(ctx context.Context, cfg client.Config, timeout time.Duration) (client.Client, error) {
	if cfg.Address == "" {
		return nil, fmt.Errorf("milvus address is required")
	}

	dialCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		dialCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	milvusClient, err := client.NewClient(dialCtx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Milvus at %s: %w", cfg.Address, err)
	}
	return milvusClient, nil
}

// EnsureCollectionLoaded ensures collection lifecycle is handled consistently:
// create when missing and load for queries.
func EnsureCollectionLoaded(
	ctx context.Context,
	milvusClient LifecycleClient,
	collectionName string,
	createIfMissing func(context.Context) error,
) error {
	return EnsureCollectionLoadedWithHooks(ctx, milvusClient, collectionName, createIfMissing, nil)
}

// EnsureCollectionLoadedWithHooks extends EnsureCollectionLoaded with an optional
// hook when the collection already exists.
func EnsureCollectionLoadedWithHooks(
	ctx context.Context,
	milvusClient LifecycleClient,
	collectionName string,
	createIfMissing func(context.Context) error,
	onExists func(context.Context) error,
) error {
	if createIfMissing == nil {
		return fmt.Errorf("createIfMissing callback must not be nil")
	}

	exists, err := milvusClient.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection %s: %w", collectionName, err)
	}

	if exists && onExists != nil {
		if err := onExists(ctx); err != nil {
			return fmt.Errorf("failed to validate existing collection %s: %w", collectionName, err)
		}
	}
	if !exists {
		if err := createIfMissing(ctx); err != nil {
			return fmt.Errorf("failed to create collection %s: %w", collectionName, err)
		}
	}

	if err := milvusClient.LoadCollection(ctx, collectionName, false); err != nil {
		return fmt.Errorf("failed to load collection %s: %w", collectionName, err)
	}
	return nil
}

// Retry retries lifecycle operations with bounded exponential backoff.
func Retry(
	ctx context.Context,
	attempts int,
	baseDelay time.Duration,
	opName string,
	op func(context.Context) error,
) error {
	if opName == "" {
		opName = "operation"
	}
	if op == nil {
		return fmt.Errorf("%s function is nil", opName)
	}

	if attempts <= 0 {
		attempts = 1
	}
	if baseDelay <= 0 {
		baseDelay = 200 * time.Millisecond
	}

	var lastErr error
	for i := 0; i < attempts; i++ {
		if err := op(ctx); err == nil {
			return nil
		} else {
			lastErr = err
		}

		if i == attempts-1 {
			break
		}

		delay := baseDelay * time.Duration(1<<i)
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s aborted: %w", opName, ctx.Err())
		case <-time.After(delay):
		}
	}

	return fmt.Errorf("%s failed after %d attempt(s): %w", opName, attempts, lastErr)
}
