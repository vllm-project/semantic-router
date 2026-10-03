package modelruntime

import (
	"context"
	"errors"
	"fmt"
	"time"
)

const (
	firstPollInterval = 250 * time.Millisecond
	maxPollInterval   = 5 * time.Second
)

// Eventually calls check until it succeeds, the timeout passes or ctx ends.
// The interval doubles from 250 ms to 5 s, so a ready system answers at once
// and a slow one is not polled hard. The last error explains a timeout; a
// check returns Stop(err) when waiting longer cannot help.
func Eventually(ctx context.Context, timeout time.Duration, check func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	interval := firstPollInterval
	for {
		err := check(ctx)
		if err == nil {
			return nil
		}
		var stop stopError
		if errors.As(err, &stop) {
			return stop.err
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("not reached within %s: %w", timeout, err)
		case <-timer.C:
		}
		if interval *= 2; interval > maxPollInterval {
			interval = maxPollInterval
		}
	}
}

// Stop marks a check's error as final, so Eventually returns it at once.
func Stop(err error) error { return stopError{err: err} }

type stopError struct{ err error }

func (e stopError) Error() string { return e.err.Error() }

func (e stopError) Unwrap() error { return e.err }
