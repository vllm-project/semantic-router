package upstream

import "time"

// Clock drives the background schedules: outlier sweeps, health checks and
// retry back-off. Tests inject a manual clock to step through them.
type Clock interface {
	Now() time.Time
	// AfterFunc calls f in its own goroutine once d has elapsed.
	AfterFunc(d time.Duration, f func()) Timer
}

// Timer is a scheduled call that can be cancelled.
type Timer interface {
	// Stop prevents the call if it has not run yet.
	Stop() bool
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

func (realClock) AfterFunc(d time.Duration, f func()) Timer { return time.AfterFunc(d, f) }
