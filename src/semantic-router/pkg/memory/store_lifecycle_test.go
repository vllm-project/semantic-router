package memory

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
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

func TestStoreLifecycleCloseCancelsAndDrainsWork(t *testing.T) {
	var life storeLifecycle
	opCtx, release, err := life.begin(context.Background(), true)
	require.NoError(t, err)
	jobStarted := make(chan struct{})
	var jobStopped, released atomic.Bool
	life.goBackground(func(ctx context.Context) {
		close(jobStarted)
		<-ctx.Done()
		jobStopped.Store(true)
	})
	<-jobStarted

	closed := make(chan error, 1)
	go func() { closed <- life.close("test", func() error { released.Store(true); return nil }) }()
	select {
	case <-opCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("Close must cancel admitted operations")
	}
	_, _, err = life.begin(context.Background(), true)
	require.ErrorIs(t, err, ErrStoreClosed)
	require.False(t, released.Load(), "the client stays alive while an operation is admitted")

	release()
	select {
	case err := <-closed:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("Close must return once admitted work drains")
	}
	require.True(t, released.Load())
	require.True(t, jobStopped.Load(), "Close cancels and waits for background jobs")

	var ran atomic.Bool
	life.goBackground(func(context.Context) { ran.Store(true) })
	require.Never(t, ran.Load, 20*time.Millisecond, 5*time.Millisecond, "no background work starts after Close")
	require.NoError(t, life.close("test", func() error { t.Fatal("release runs once"); return nil }))
}

func TestStoreLifecycleReleasesClientAfterSlowWork(t *testing.T) {
	life := storeLifecycle{closeWait: 20 * time.Millisecond}
	_, release, err := life.begin(context.Background(), true)
	require.NoError(t, err)

	var released atomic.Bool
	require.NoError(t, life.close("test", func() error { released.Store(true); return nil }))
	require.False(t, released.Load(), "Close returns after its wait but keeps the client for the running operation")
	release()
	require.Eventually(t, released.Load, time.Second, 5*time.Millisecond, "the client is released once the operation returns")
}

// A Qdrant Retrieve stuck in embedding past the close wait must not reach a
// released client; the Qdrant client panics when used after its own Close.
func TestQdrantSlowRetrieveDoesNotPanicAfterClose(t *testing.T) {
	entered, unblock := make(chan struct{}), make(chan struct{})
	provider, err := embedding.NewFuncProvider("test", 4, func(context.Context, string) ([]float32, error) {
		close(entered)
		<-unblock
		return []float32{1, 0, 0, 0}, nil
	})
	require.NoError(t, err)
	store := &QdrantStore{
		client:          unreachableQdrantClient(t),
		collectionName:  "c",
		enabled:         true,
		embeddingConfig: EmbeddingConfig{Provider: provider},
	}
	store.life.closeWait = 20 * time.Millisecond

	done := make(chan error, 1)
	go func() {
		_, retrieveErr := store.Retrieve(context.Background(), RetrieveOptions{Query: "q", UserID: "u"})
		done <- retrieveErr
	}()
	<-entered
	require.NoError(t, store.Close())
	close(unblock)
	require.Error(t, <-done, "the retrieval fails cleanly instead of panicking")
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
