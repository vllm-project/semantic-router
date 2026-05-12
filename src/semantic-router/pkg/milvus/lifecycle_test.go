package milvus

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

type fakeLifecycleClient struct {
	hasCalls         int
	loadCalls        int
	hasCollectionFn  func(context.Context, string) (bool, error)
	loadCollectionFn func(context.Context, string, bool, ...client.LoadCollectionOption) error
}

func (f *fakeLifecycleClient) HasCollection(ctx context.Context, collectionName string) (bool, error) {
	f.hasCalls++
	if f.hasCollectionFn != nil {
		return f.hasCollectionFn(ctx, collectionName)
	}
	return false, nil
}

func (f *fakeLifecycleClient) LoadCollection(ctx context.Context, collectionName string, async bool, opts ...client.LoadCollectionOption) error {
	f.loadCalls++
	if f.loadCollectionFn != nil {
		return f.loadCollectionFn(ctx, collectionName, async, opts...)
	}
	return nil
}

func TestRetry_SucceedsFirstAttempt(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	calls := 0
	err := Retry(ctx, 3, time.Millisecond, "op", func(context.Context) error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("expected nil, got %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}
}

func TestRetry_EventuallySucceeds(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	attempt := 0
	err := Retry(ctx, 4, time.Millisecond, "op", func(context.Context) error {
		attempt++
		if attempt < 3 {
			return errors.New("transient")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("expected nil after retries: %v", err)
	}
	if attempt != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempt)
	}
}

func TestRetry_AllAttemptsFail(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	want := errors.New("permanent")
	err := Retry(ctx, 2, time.Millisecond, "op", func(context.Context) error {
		return want
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRetry_ContextCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Retry(ctx, 3, time.Millisecond, "op", func(context.Context) error {
		return errors.New("fail")
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEnsureCollectionLoaded_CreatesAndLoads(t *testing.T) {
	t.Parallel()

	f := &fakeLifecycleClient{}
	col := "demo"
	createCalls := 0

	err := EnsureCollectionLoaded(context.Background(), f, col, func(innerCtx context.Context) error {
		createCalls++
		return nil
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if createCalls != 1 {
		t.Fatalf("create once: got %d", createCalls)
	}
	if f.hasCalls != 1 {
		t.Fatalf("HasCollection calls: got %d", f.hasCalls)
	}
	if f.loadCalls != 1 {
		t.Fatalf("expected LoadCollection once, got %d", f.loadCalls)
	}
}

func TestEnsureCollectionLoaded_ExistingLoadsOnly(t *testing.T) {
	t.Parallel()

	f := &fakeLifecycleClient{
		hasCollectionFn: func(context.Context, string) (bool, error) {
			return true, nil
		},
	}
	createCalls := 0

	err := EnsureCollectionLoaded(context.Background(), f, "c", func(context.Context) error {
		createCalls++
		return nil
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	if createCalls != 0 {
		t.Fatalf("expected zero create callbacks, got %d", createCalls)
	}
	if f.loadCalls != 1 {
		t.Fatalf("LoadCollection: got %d", f.loadCalls)
	}
}

func TestEnsureCollectionLoadedWithHooks_OnExisting(t *testing.T) {
	t.Parallel()

	f := &fakeLifecycleClient{
		hasCollectionFn: func(context.Context, string) (bool, error) {
			return true, nil
		},
	}
	onExistsRuns := 0

	err := EnsureCollectionLoadedWithHooks(
		context.Background(),
		f,
		"t",
		func(context.Context) error { return errors.New("should not create") },
		func(context.Context) error {
			onExistsRuns++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if onExistsRuns != 1 {
		t.Fatalf("onExistsRuns=%d", onExistsRuns)
	}
}
