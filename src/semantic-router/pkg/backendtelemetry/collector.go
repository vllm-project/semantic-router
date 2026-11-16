package backendtelemetry

import (
	"context"
	"errors"
	"sync"
	"time"
)

// CollectorOptions configures a Collector.
type CollectorOptions struct {
	Adapters []Adapter
	Store    *Store
	// Interval is the time between collection rounds.
	Interval time.Duration
	// Report receives every adapter's result for each round: the
	// observations it built and the joined errors of the targets it could
	// not read. It must not block. Nil discards results.
	Report func(engine Engine, observations []Observation, err error)
}

// Collector runs its adapters off the request path, once at start and then
// every interval, and records their observations in the store. A target that
// fails a round simply records nothing, so its last observation ages out.
type Collector struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// StartCollector starts collecting. A collector without adapters does
// nothing until closed.
func StartCollector(opts CollectorOptions) *Collector {
	ctx, cancel := context.WithCancel(context.Background())
	c := &Collector{cancel: cancel, done: make(chan struct{})}
	if len(opts.Adapters) == 0 || opts.Store == nil || opts.Interval <= 0 {
		close(c.done)
		return c
	}
	go c.run(ctx, opts)
	return c
}

// idleCloser is an adapter that keeps connections between rounds.
type idleCloser interface {
	CloseIdleConnections()
}

func (c *Collector) run(ctx context.Context, opts CollectorOptions) {
	defer close(c.done)
	// Runs after the last round's Collect calls return, even when a bounded
	// Close stopped waiting, so no in-flight scrape loses its connection.
	defer func() {
		for _, adapter := range opts.Adapters {
			if closer, ok := adapter.(idleCloser); ok {
				closer.CloseIdleConnections()
			}
		}
	}()
	ticker := time.NewTicker(opts.Interval)
	defer ticker.Stop()
	for {
		collectRound(ctx, opts)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func collectRound(ctx context.Context, opts CollectorOptions) {
	var wg sync.WaitGroup
	for _, adapter := range opts.Adapters {
		wg.Add(1)
		go func() {
			defer wg.Done()
			observations, err := adapter.Collect(ctx)
			if ctx.Err() != nil {
				return
			}
			err = errors.Join(err, opts.Store.UpsertMany(observations))
			if opts.Report != nil {
				opts.Report(adapter.Engine(), observations, err)
			}
		}()
	}
	wg.Wait()
}

// Close stops collection and waits for an in-flight round to end, or for ctx.
func (c *Collector) Close(ctx context.Context) error {
	c.cancel()
	select {
	case <-c.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
