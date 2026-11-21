package cache

import (
	"context"
	"time"
)

const (
	minExpiredRowSweepInterval = time.Minute
	maxExpiredRowSweepInterval = time.Hour
	expiredRowSweepTimeout     = 2 * time.Minute
)

// Reads already treat an expired row as a miss, so the sweep only bounds how
// long such rows keep storage; a pass at least hourly covers per-entry TTLs
// set by decisions even when the store default is zero.
func expiredRowSweepInterval(defaultTTLSeconds int) time.Duration {
	if defaultTTLSeconds <= 0 {
		return maxExpiredRowSweepInterval
	}
	interval := time.Duration(defaultTTLSeconds) * time.Second
	return min(max(interval, minExpiredRowSweepInterval), maxExpiredRowSweepInterval)
}

type expiredRowSweeper struct {
	cancel context.CancelFunc
	done   chan struct{}
}

func startExpiredRowSweeper(interval time.Duration, sweep func(context.Context) error) *expiredRowSweeper {
	ctx, cancel := context.WithCancel(context.Background())
	sweeper := &expiredRowSweeper{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(sweeper.done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				passCtx, cancelPass := context.WithTimeout(ctx, expiredRowSweepTimeout)
				_ = sweep(passCtx)
				cancelPass()
			}
		}
	}()
	return sweeper
}

// stop cancels any in-flight pass and waits for the goroutine to exit.
func (s *expiredRowSweeper) stop() {
	if s == nil {
		return
	}
	s.cancel()
	<-s.done
}
