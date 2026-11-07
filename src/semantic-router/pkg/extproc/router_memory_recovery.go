package extproc

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type memoryStoreFactory func(context.Context, *config.RouterConfig, ...*embedding.Set) (memory.Store, error)

type memoryReconnectPolicy struct {
	initialBackoff time.Duration
	maxBackoff     time.Duration
	connectTimeout time.Duration
}

var defaultMemoryReconnectPolicy = memoryReconnectPolicy{
	initialBackoff: time.Second,
	maxBackoff:     30 * time.Second,
	connectTimeout: 5 * time.Second,
}

// memoryStoreRuntime owns the store slot and its reconnect loop for exactly one
// router generation. Close cancels an in-flight connection attempt, waits for
// the loop, and then closes a recovered or initially connected store.
type memoryStoreRuntime struct {
	slot   *memory.StoreSlot
	ctx    context.Context
	cancel context.CancelFunc

	mu      sync.Mutex
	started bool
	closed  bool
	loop    sync.WaitGroup

	closeOnce sync.Once
	closeErr  error
}

func newMemoryStoreRuntime(store memory.Store) *memoryStoreRuntime {
	ctx, cancel := context.WithCancel(context.Background())
	return &memoryStoreRuntime{
		slot:   memory.NewStoreSlot(store),
		ctx:    ctx,
		cancel: cancel,
	}
}

func memoryStoreSlot(runtime *memoryStoreRuntime) *memory.StoreSlot {
	if runtime == nil {
		return nil
	}
	return runtime.slot
}

func (r *memoryStoreRuntime) startRecovery(
	cfg *config.RouterConfig,
	sets []*embedding.Set,
	factory memoryStoreFactory,
	policy memoryReconnectPolicy,
) {
	if r == nil || factory == nil || r.slot.Load() != nil {
		return
	}
	if policy.initialBackoff <= 0 {
		policy.initialBackoff = defaultMemoryReconnectPolicy.initialBackoff
	}
	if policy.maxBackoff < policy.initialBackoff {
		policy.maxBackoff = policy.initialBackoff
	}
	if policy.connectTimeout <= 0 {
		policy.connectTimeout = defaultMemoryReconnectPolicy.connectTimeout
	}

	r.mu.Lock()
	if r.started || r.closed {
		r.mu.Unlock()
		return
	}
	r.started = true
	r.loop.Add(1)
	r.mu.Unlock()

	goSafely("memory-store-reconnect", func() {
		defer r.loop.Done()
		r.reconnect(cfg, sets, factory, policy)
	})
}

func (r *memoryStoreRuntime) reconnect(
	cfg *config.RouterConfig,
	sets []*embedding.Set,
	factory memoryStoreFactory,
	policy memoryReconnectPolicy,
) {
	backoff := policy.initialBackoff
	for {
		timer := time.NewTimer(backoff)
		select {
		case <-r.ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}

		attemptCtx, cancel := context.WithTimeout(r.ctx, policy.connectTimeout)
		store, err := factory(attemptCtx, cfg, sets...)
		cancel()
		if err == nil && store == nil {
			err = errors.New("memory store factory returned a nil store")
		}
		if err == nil {
			if r.ctx.Err() != nil || !r.slot.Install(store) {
				_ = store.Close()
				return
			}
			logging.ComponentEvent("memory", "memory_store_reconnected", map[string]interface{}{
				"backend": memoryBackendName(cfg),
			})
			return
		}
		if r.ctx.Err() != nil {
			return
		}
		nextBackoff := backoff
		if backoff < policy.maxBackoff {
			nextBackoff *= 2
			if nextBackoff > policy.maxBackoff {
				nextBackoff = policy.maxBackoff
			}
		}
		logging.ComponentWarnEvent("memory", "memory_store_reconnect_failed", map[string]interface{}{
			"backend":        memoryBackendName(cfg),
			"retry_after_ms": nextBackoff.Milliseconds(),
			"error":          err.Error(),
		})
		backoff = nextBackoff
	}
}

func (r *memoryStoreRuntime) Close() error {
	if r == nil {
		return nil
	}
	r.closeOnce.Do(func() {
		r.mu.Lock()
		r.closed = true
		r.cancel()
		r.mu.Unlock()
		r.loop.Wait()
		r.closeErr = r.slot.Close()
	})
	return r.closeErr
}

func memoryBackendName(cfg *config.RouterConfig) string {
	if cfg != nil && cfg.Memory.Backend != "" {
		return cfg.Memory.Backend
	}
	return "milvus"
}
