package cache

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExpiredRowSweepInterval(t *testing.T) {
	tests := []struct {
		name              string
		defaultTTLSeconds int
		want              time.Duration
	}{
		{name: "no default TTL sweeps hourly", defaultTTLSeconds: 0, want: time.Hour},
		{name: "negative TTL sweeps hourly", defaultTTLSeconds: -1, want: time.Hour},
		{name: "short TTL is clamped to a minute", defaultTTLSeconds: 5, want: time.Minute},
		{name: "TTL inside the bounds is used", defaultTTLSeconds: 600, want: 10 * time.Minute},
		{name: "long TTL is clamped to an hour", defaultTTLSeconds: 86400, want: time.Hour},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, expiredRowSweepInterval(tt.defaultTTLSeconds))
		})
	}
}

func TestExpiredRowSweeperRepeatsUntilStopped(t *testing.T) {
	var passes atomic.Int32
	sweeper := startExpiredRowSweeper(5*time.Millisecond, func(context.Context) error {
		passes.Add(1)
		return nil
	})
	require.Eventually(t, func() bool { return passes.Load() >= 2 }, time.Second, time.Millisecond)

	sweeper.stop()
	stopped := passes.Load()
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, stopped, passes.Load())
}

func TestExpiredRowSweeperStopCancelsInFlightPass(t *testing.T) {
	started := make(chan struct{})
	passErr := make(chan error, 1)
	sweeper := startExpiredRowSweeper(time.Millisecond, func(ctx context.Context) error {
		close(started)
		<-ctx.Done()
		passErr <- ctx.Err()
		return ctx.Err()
	})
	<-started

	sweeper.stop()
	assert.ErrorIs(t, <-passErr, context.Canceled)
}

func TestExpiredRowSweeperStopIsNilSafe(t *testing.T) {
	var sweeper *expiredRowSweeper
	assert.NotPanics(t, sweeper.stop)
}
