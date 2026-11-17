package extproc

import (
	"context"
	"sync"
	"time"
)

const (
	// replayReadCacheTTL bounds how stale a cached replay read may be.
	replayReadCacheTTL = 30 * time.Second
	// replayReadCacheEntries bounds the filter combinations one cache keeps.
	replayReadCacheEntries = 32
)

// replayReadCache keeps one router generation's recent answers to a read that
// scans the whole replay history, by filter, and runs at most one such scan per
// filter at a time: a caller that asks while it runs waits for its answer. The
// scan stops once every caller waiting for it has left, and a failed scan is
// not kept.
type replayReadCache[V any] struct {
	mu      sync.Mutex
	now     func() time.Time
	entries map[string]replayReadCacheEntry[V]
	flights map[string]*replayReadFlight[V]
}

type replayReadCacheEntry[V any] struct {
	value   V
	expires time.Time
}

type replayReadFlight[V any] struct {
	done    chan struct{}
	value   V
	err     error
	waiters int
	cancel  context.CancelFunc
}

// get answers from the cache, from the scan already running for key, or from
// a new scan. Callers must not modify the value: other callers share it.
func (c *replayReadCache[V]) get(ctx context.Context, key string, scan func(context.Context) (V, error)) (V, error) {
	c.mu.Lock()
	if entry, ok := c.entries[key]; ok {
		if c.clock().Before(entry.expires) {
			c.mu.Unlock()
			return entry.value, nil
		}
		delete(c.entries, key)
	}
	flight := c.flights[key]
	if flight == nil {
		if c.flights == nil {
			c.flights = make(map[string]*replayReadFlight[V])
		}
		scanCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
		flight = &replayReadFlight[V]{done: make(chan struct{}), cancel: cancel}
		c.flights[key] = flight
		go c.run(scanCtx, key, flight, scan)
	}
	flight.waiters++
	c.mu.Unlock()

	select {
	case <-flight.done:
		return flight.value, flight.err
	case <-ctx.Done():
		c.leave(key, flight)
		var zero V
		return zero, ctx.Err()
	}
}

// leave stops the scan when its last waiting caller leaves; a caller that
// asks afterwards starts a new one.
func (c *replayReadCache[V]) leave(key string, flight *replayReadFlight[V]) {
	c.mu.Lock()
	defer c.mu.Unlock()
	flight.waiters--
	if flight.waiters > 0 {
		return
	}
	flight.cancel()
	if c.flights[key] == flight {
		delete(c.flights, key)
	}
}

func (c *replayReadCache[V]) run(ctx context.Context, key string, flight *replayReadFlight[V], scan func(context.Context) (V, error)) {
	value, err := scan(ctx)
	flight.cancel()
	c.mu.Lock()
	flight.value, flight.err = value, err
	if c.flights[key] == flight {
		delete(c.flights, key)
		if err == nil {
			c.keep(key, value)
		}
	}
	c.mu.Unlock()
	close(flight.done)
}

func (c *replayReadCache[V]) keep(key string, value V) {
	now := c.clock()
	if c.entries == nil {
		c.entries = make(map[string]replayReadCacheEntry[V])
	}
	for stale, entry := range c.entries {
		if !now.Before(entry.expires) {
			delete(c.entries, stale)
		}
	}
	if len(c.entries) >= replayReadCacheEntries {
		clear(c.entries)
	}
	c.entries[key] = replayReadCacheEntry[V]{value: value, expires: now.Add(replayReadCacheTTL)}
}

func (c *replayReadCache[V]) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}
