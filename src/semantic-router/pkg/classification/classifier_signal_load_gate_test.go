package classification

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSignalLoadGateLimitsConcurrentEvaluations(t *testing.T) {
	t.Parallel()

	gate := newSignalLoadGate(config.BatchClassificationConfig{
		ConcurrencyThreshold: 2,
		MaxConcurrency:       3,
	})
	if gate == nil {
		t.Fatal("expected load gate to be configured")
	}

	var active int64
	var maxSeen int64
	start := make(chan struct{})

	var wg sync.WaitGroup
	for range 6 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start

			release := gate.enter()
			current := atomic.AddInt64(&active, 1)
			updateMaxInt64(&maxSeen, current)
			time.Sleep(20 * time.Millisecond)
			atomic.AddInt64(&active, -1)
			release()
		}()
	}

	close(start)
	wg.Wait()

	if got := atomic.LoadInt64(&maxSeen); got > 3 {
		t.Fatalf("expected at most 3 concurrent evaluations, got %d", got)
	}
}

func TestSignalLoadGateDisabledWithoutMaxConcurrency(t *testing.T) {
	t.Parallel()

	if gate := newSignalLoadGate(config.BatchClassificationConfig{}); gate != nil {
		t.Fatal("expected nil gate when max concurrency is unset")
	}
}

func updateMaxInt64(target *int64, candidate int64) {
	for {
		current := atomic.LoadInt64(target)
		if candidate <= current {
			return
		}
		if atomic.CompareAndSwapInt64(target, current, candidate) {
			return
		}
	}
}
