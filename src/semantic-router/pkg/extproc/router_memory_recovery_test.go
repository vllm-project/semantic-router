package extproc

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type recoveryMemoryStore struct {
	*memory.InMemoryStore
	closeCalls atomic.Int32
}

func newRecoveryMemoryStore() *recoveryMemoryStore {
	return &recoveryMemoryStore{InMemoryStore: memory.NewInMemoryStore()}
}

func (s *recoveryMemoryStore) Close() error {
	s.closeCalls.Add(1)
	return s.InMemoryStore.Close()
}

func fastMemoryReconnectPolicy() memoryReconnectPolicy {
	return memoryReconnectPolicy{
		initialBackoff: time.Millisecond,
		maxBackoff:     4 * time.Millisecond,
		connectTimeout: 100 * time.Millisecond,
	}
}

func TestMemoryStoreRecoveryPublishesToRouterAndRegistry(t *testing.T) {
	runtime := newMemoryStoreRuntime(nil)
	t.Cleanup(func() { require.NoError(t, runtime.Close()) })

	recovered := newRecoveryMemoryStore()
	var attempts atomic.Int32
	factory := func(context.Context, *config.RouterConfig, ...*embedding.Set) (memory.Store, error) {
		if attempts.Add(1) < 3 {
			return nil, errors.New("backend is starting")
		}
		return recovered, nil
	}
	router := &OpenAIRouter{memoryStore: runtime.slot}
	registry := routerruntime.NewRegistry(&config.RouterConfig{Memory: config.MemoryConfig{Enabled: true}})
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
		MemoryStoreSlot: runtime.slot,
		MemoryStatus: routerruntime.MemoryStatus{
			Reason:  routerruntime.MemoryReasonBackendUnavailable,
			Backend: "milvus",
		},
		AcquireClassification: func() (func(), bool) { return func() {}, true },
	})

	runtime.startRecovery(&config.RouterConfig{Memory: config.MemoryConfig{Backend: "milvus"}}, nil, factory, fastMemoryReconnectPolicy())

	require.Eventually(t, func() bool {
		return router.getMemoryStore() == recovered && registry.MemoryStore() == recovered
	}, time.Second, time.Millisecond)
	require.Equal(t, int32(3), attempts.Load())
	require.Empty(t, registry.MemoryStatus().Reason)

	store, release, ok := registry.AcquireMemoryStore()
	require.True(t, ok)
	require.Same(t, recovered, store)
	require.NoError(t, store.Store(context.Background(), &memory.Memory{
		ID: "recovered-memory", UserID: "user-1", Content: "backend recovered", Embedding: []float32{1},
	}))
	stored, err := store.Get(context.Background(), "recovered-memory")
	require.NoError(t, err)
	require.Equal(t, "backend recovered", stored.Content)
	release()

	require.NoError(t, runtime.Close())
	require.Equal(t, int32(1), recovered.closeCalls.Load())
}

func TestMemoryStoreRecoveryStopsPromptlyOnReloadAndShutdown(t *testing.T) {
	for _, action := range []string{"reload", "shutdown"} {
		t.Run(action, func(t *testing.T) {
			runtime := newMemoryStoreRuntime(nil)
			resources := newResourceScope()
			resources.add(runtime.Close)
			service := NewRouterService(&OpenAIRouter{memoryStore: runtime.slot, resources: resources})
			t.Cleanup(func() { require.NoError(t, service.Shutdown(context.Background())) })

			attempted := make(chan struct{})
			attemptDone := make(chan struct{})
			var attempts atomic.Int32
			factory := func(ctx context.Context, _ *config.RouterConfig, _ ...*embedding.Set) (memory.Store, error) {
				attempts.Add(1)
				close(attempted)
				<-ctx.Done()
				close(attemptDone)
				return nil, ctx.Err()
			}
			policy := fastMemoryReconnectPolicy()
			policy.connectTimeout = time.Minute
			runtime.startRecovery(&config.RouterConfig{}, nil, factory, policy)

			select {
			case <-attempted:
			case <-time.After(time.Second):
				t.Fatal("reconnect attempt did not start")
			}
			if action == "reload" {
				require.NoError(t, service.Swap(&OpenAIRouter{resources: newResourceScope()}, nil))
			} else {
				require.NoError(t, service.Shutdown(context.Background()))
			}
			select {
			case <-attemptDone:
			case <-time.After(time.Second):
				t.Fatal("generation retirement did not cancel the reconnect attempt")
			}
			time.Sleep(10 * time.Millisecond)
			require.Equal(t, int32(1), attempts.Load())
			require.Nil(t, runtime.slot.Load())
		})
	}
}

func TestRetiredMemoryRecoveryCannotReplaceNewRegistryGeneration(t *testing.T) {
	oldRuntime := newMemoryStoreRuntime(nil)
	t.Cleanup(func() { require.NoError(t, oldRuntime.Close()) })
	oldRecovered := newRecoveryMemoryStore()
	allowSuccess := make(chan struct{})
	factory := func(ctx context.Context, _ *config.RouterConfig, _ ...*embedding.Set) (memory.Store, error) {
		select {
		case <-allowSuccess:
			return oldRecovered, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	oldRuntime.startRecovery(&config.RouterConfig{}, nil, factory, fastMemoryReconnectPolicy())

	registry := routerruntime.NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{MemoryStoreSlot: oldRuntime.slot})
	newStore := newRecoveryMemoryStore()
	newSlot := memory.NewStoreSlot(newStore)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{MemoryStoreSlot: newSlot})
	close(allowSuccess)

	require.Eventually(t, func() bool { return oldRuntime.slot.Load() == oldRecovered }, time.Second, time.Millisecond)
	require.Same(t, newStore, registry.MemoryStore())
	require.NoError(t, oldRuntime.Close())
	require.Equal(t, int32(1), oldRecovered.closeCalls.Load())
	require.Same(t, newStore, registry.MemoryStore())
	require.NoError(t, newSlot.Close())
}
