package memory

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
)

type countingCloseMilvus struct {
	*MockMilvusClient
	closes atomic.Int32
}

func (c *countingCloseMilvus) Close() error {
	c.closes.Add(1)
	return nil
}

// unreachableQdrantClient connects lazily, so no Qdrant server is needed.
func unreachableQdrantClient(t *testing.T) *qdrant.Client {
	t.Helper()
	client, err := qdrant.NewClient(&qdrant.Config{Host: "127.0.0.1", Port: 1, SkipCompatibilityCheck: true})
	require.NoError(t, err)
	return client
}

// requireStoreRefuses calls every Store method and expects want from each.
func requireStoreRefuses(t *testing.T, store Store, want error) {
	t.Helper()
	ctx := context.Background()
	mem := &Memory{ID: "m1", UserID: "u1", Content: "x", Embedding: []float32{1}}
	_, retrieveErr := store.Retrieve(ctx, RetrieveOptions{Query: "x", UserID: "u1"})
	_, getErr := store.Get(ctx, "m1")
	_, listErr := store.List(ctx, ListOptions{UserID: "u1"})
	for name, err := range map[string]error{
		"Store":           store.Store(ctx, mem),
		"Retrieve":        retrieveErr,
		"Get":             getErr,
		"Update":          store.Update(ctx, "m1", mem),
		"List":            listErr,
		"Forget":          store.Forget(ctx, "m1"),
		"ForgetByScope":   store.ForgetByScope(ctx, MemoryScope{UserID: "u1"}),
		"CheckConnection": store.CheckConnection(ctx),
	} {
		require.ErrorIs(t, err, want, name)
	}
	require.False(t, store.IsEnabled())
	require.NoError(t, store.Close(), "Close is idempotent")
}

func TestDisabledStoresReturnErrStoreDisabled(t *testing.T) {
	milvus, err := NewMilvusStore(MilvusStoreOptions{Enabled: false})
	require.NoError(t, err)
	qdrantStore, err := NewQdrantStore(QdrantStoreOptions{Enabled: false})
	require.NoError(t, err)
	valkey, err := NewValkeyStore(ValkeyStoreOptions{Enabled: false})
	require.NoError(t, err)

	for name, store := range map[string]Store{"milvus": milvus, "qdrant": qdrantStore, "valkey": valkey} {
		t.Run(name, func(t *testing.T) { requireStoreRefuses(t, store, ErrStoreDisabled) })
	}
}

func TestClosedStoresReturnErrStoreClosed(t *testing.T) {
	milvus, _ := setupTestStore()
	inMemory := NewInMemoryStore()
	qdrantStore := &QdrantStore{client: unreachableQdrantClient(t), collectionName: "c", enabled: true}

	for name, store := range map[string]Store{"milvus": milvus, "qdrant": qdrantStore, "in-memory": inMemory} {
		t.Run(name, func(t *testing.T) {
			require.True(t, store.IsEnabled())
			require.NoError(t, store.Close())
			requireStoreRefuses(t, store, ErrStoreClosed)
		})
	}
}

func TestMilvusCloseClosesClientOnce(t *testing.T) {
	store, mock := setupTestStore()
	client := &countingCloseMilvus{MockMilvusClient: mock}
	store.client = client

	require.NoError(t, store.Close())
	require.NoError(t, store.Close())
	require.Equal(t, int32(1), client.closes.Load(), "the store owns its client and closes it exactly once")
}

func TestStoreLifecycleCloseWaitsForWork(t *testing.T) {
	var life storeLifecycle
	release, err := life.begin(true)
	require.NoError(t, err)
	jobStarted, jobStopped := make(chan struct{}), atomic.Bool{}
	life.goBackground(func(ctx context.Context) {
		close(jobStarted)
		<-ctx.Done()
		jobStopped.Store(true)
	})
	<-jobStarted

	closed := make(chan struct{})
	go func() {
		life.close("test")
		close(closed)
	}()
	require.Eventually(t, func() bool {
		_, beginErr := life.begin(true)
		return beginErr == ErrStoreClosed
	}, time.Second, time.Millisecond, "no new work is admitted once Close starts")
	require.Never(t, func() bool {
		select {
		case <-closed:
			return true
		default:
			return false
		}
	}, 50*time.Millisecond, 5*time.Millisecond, "Close waits for the in-flight operation")

	release()
	require.Eventually(t, func() bool {
		select {
		case <-closed:
			return true
		default:
			return false
		}
	}, time.Second, time.Millisecond)
	require.True(t, jobStopped.Load(), "Close cancels and waits for background jobs")

	ran := atomic.Bool{}
	life.goBackground(func(context.Context) { ran.Store(true) })
	require.Never(t, ran.Load, 20*time.Millisecond, 5*time.Millisecond, "no background work starts after Close")
	require.False(t, life.close("test"), "a second close is a no-op")
}

func TestInMemoryStoreRacingCloseDoesNotPanic(t *testing.T) {
	for i := 0; i < 200; i++ {
		store := NewInMemoryStore()
		done := make(chan struct{})
		go func() {
			defer close(done)
			_ = store.Store(context.Background(), &Memory{ID: "m", UserID: "u", Embedding: []float32{1}})
		}()
		require.NoError(t, store.Close())
		<-done
	}
}
