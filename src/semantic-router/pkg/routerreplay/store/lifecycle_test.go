package store

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestStorageLifecycleDrainsAcceptedMutationAndRejectsNewWork(t *testing.T) {
	lifecycle := newStorageLifecycle()
	release, err := lifecycle.beginMutation()
	if err != nil {
		t.Fatalf("beginMutation() error = %v", err)
	}

	var closeCalls atomic.Int32
	closed := make(chan error, 1)
	go func() {
		closed <- lifecycle.close(func() error {
			closeCalls.Add(1)
			return nil
		})
	}()
	select {
	case err := <-closed:
		t.Fatalf("close returned before accepted mutation drained: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	release()
	if err := <-closed; err != nil {
		t.Fatalf("close() error = %v", err)
	}
	if closeCalls.Load() != 1 {
		t.Fatalf("close callback calls = %d, want 1", closeCalls.Load())
	}
	if _, err := lifecycle.beginMutation(); !errors.Is(err, ErrStorageClosed) {
		t.Fatalf("beginMutation() after close error = %v, want ErrStorageClosed", err)
	}
	if err := lifecycle.close(func() error {
		closeCalls.Add(1)
		return nil
	}); err != nil {
		t.Fatalf("second close() error = %v", err)
	}
	if closeCalls.Load() != 1 {
		t.Fatalf("close callback calls after second Close = %d, want 1", closeCalls.Load())
	}
}

func TestStorageLifecycleBoundsStalledCloseCallback(t *testing.T) {
	lifecycle := newStorageLifecycle()
	started := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := lifecycle.closeContext(ctx, func() error {
		close(started)
		<-release
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("close() error = %v, want context deadline exceeded", err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("close callback was not attempted")
	}
}

func TestBackendStoresRejectReadsAfterLifecycleClose(t *testing.T) {
	tests := []struct {
		name string
		get  func() error
		list func() error
	}{
		{
			name: "milvus",
			get: func() error {
				store := &MilvusStore{lifecycle: closedStorageLifecycle(), asyncWrites: true}
				_, _, err := store.Get(context.Background(), "replay-1")
				return err
			},
			list: func() error {
				store := &MilvusStore{lifecycle: closedStorageLifecycle(), asyncWrites: true}
				_, err := store.List(context.Background())
				return err
			},
		},
		{
			name: "qdrant",
			get: func() error {
				store := &QdrantStore{lifecycle: closedStorageLifecycle(), asyncWrites: true}
				_, _, err := store.Get(context.Background(), "replay-1")
				return err
			},
			list: func() error {
				store := &QdrantStore{lifecycle: closedStorageLifecycle(), asyncWrites: true}
				_, err := store.List(context.Background())
				return err
			},
		},
		{
			name: "redis",
			get: func() error {
				store := &RedisStore{lifecycle: closedStorageLifecycle()}
				_, _, err := store.Get(context.Background(), "replay-1")
				return err
			},
			list: func() error {
				store := &RedisStore{lifecycle: closedStorageLifecycle()}
				_, err := store.List(context.Background())
				return err
			},
		},
		{
			name: "postgres",
			get: func() error {
				store := &PostgresStore{lifecycle: closedStorageLifecycle()}
				_, _, err := store.Get(context.Background(), "replay-1")
				return err
			},
			list: func() error {
				store := &PostgresStore{lifecycle: closedStorageLifecycle()}
				_, err := store.List(context.Background())
				return err
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.get(); !errors.Is(err, ErrStorageClosed) {
				t.Fatalf("Get() error = %v, want ErrStorageClosed", err)
			}
			if err := test.list(); !errors.Is(err, ErrStorageClosed) {
				t.Fatalf("List() error = %v, want ErrStorageClosed", err)
			}
		})
	}
}

func closedStorageLifecycle() *storageLifecycle {
	lifecycle := newStorageLifecycle()
	_ = lifecycle.close(func() error { return nil })
	return lifecycle
}
