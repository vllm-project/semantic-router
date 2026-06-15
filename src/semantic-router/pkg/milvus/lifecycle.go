// Package milvus provides shared connection, collection ensure/load, and retry
// helpers for Milvus-backed runtime stores (memory, cache, vectorstore,
// routerreplay, etc.). Domain packages keep schema and query semantics; use
// this package for duplicated lifecycle mechanics (see GitHub issue #1601).
package milvus

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// LifecycleClient captures the subset of Milvus client lifecycle APIs reused by
// runtime stores.
type LifecycleClient interface {
	HasCollection(ctx context.Context, collectionName string) (bool, error)
	LoadCollection(ctx context.Context, collectionName string, async bool, opts ...client.LoadCollectionOption) error
}

// CollectionRetryOptions controls retry behavior for collection lifecycle
// operations during backend startup.
type CollectionRetryOptions struct {
	Attempts  int
	BaseDelay time.Duration
}

func normalizedCollectionRetryOptions(opts CollectionRetryOptions) CollectionRetryOptions {
	if opts.Attempts <= 0 {
		opts.Attempts = 3
	}
	if opts.BaseDelay <= 0 {
		opts.BaseDelay = 2 * time.Second
	}
	return opts
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

// EnsureCollectionLoadedWithRetry retries collection lifecycle operations when
// Milvus is running but still converging internal node or metadata state.
func EnsureCollectionLoadedWithRetry(
	ctx context.Context,
	milvusClient LifecycleClient,
	collectionName string,
	createIfMissing func(context.Context) error,
	opts CollectionRetryOptions,
) error {
	return EnsureCollectionLoadedWithHooksRetry(ctx, milvusClient, collectionName, createIfMissing, nil, opts)
}

// EnsureCollectionLoadedWithHooksRetry extends EnsureCollectionLoadedWithHooks
// with bounded retries for transient Milvus startup/lifecycle failures.
func EnsureCollectionLoadedWithHooksRetry(
	ctx context.Context,
	milvusClient LifecycleClient,
	collectionName string,
	createIfMissing func(context.Context) error,
	onExists func(context.Context) error,
	opts CollectionRetryOptions,
) error {
	retryOpts := normalizedCollectionRetryOptions(opts)
	return RetryIf(
		ctx,
		retryOpts.Attempts,
		retryOpts.BaseDelay,
		fmt.Sprintf("ensure/load collection %s", collectionName),
		func(innerCtx context.Context) error {
			return EnsureCollectionLoadedWithHooks(innerCtx, milvusClient, collectionName, createIfMissing, onExists)
		},
		IsTransientLifecycleError,
	)
}

// IsTransientLifecycleError identifies Milvus lifecycle errors that can appear
// while standalone Milvus has opened its gRPC port but is still reconciling
// proxy/data node metadata after a restart.
func IsTransientLifecycleError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	transientMarkers := []string{
		"context deadline exceeded",
		"connection refused",
		"grpc: the client connection is closing",
		"invalidatecollectionmetacache failed",
		"node not match",
		"server is not ready",
		"transport is closing",
		"unavailable",
	}
	for _, marker := range transientMarkers {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

// Retry retries lifecycle operations with bounded exponential backoff.
func Retry(
	ctx context.Context,
	attempts int,
	baseDelay time.Duration,
	opName string,
	op func(context.Context) error,
) error {
	return RetryIf(ctx, attempts, baseDelay, opName, op, func(error) bool { return true })
}

// RetryIf retries lifecycle operations with bounded exponential backoff while
// the supplied predicate marks failures as retryable.
func RetryIf(
	ctx context.Context,
	attempts int,
	baseDelay time.Duration,
	opName string,
	op func(context.Context) error,
	shouldRetry func(error) bool,
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
			if shouldRetry != nil && !shouldRetry(err) {
				return fmt.Errorf("%s failed with non-retryable error: %w", opName, err)
			}
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
