package upstream

import (
	"sort"
	"sync"
	"time"
)

// manualClock is a Clock that only moves when a test advances it. Advance
// runs every timer that falls due, in due order, in the calling goroutine,
// including timers those calls schedule.
type manualClock struct {
	mu     sync.Mutex
	now    time.Time
	timers []*manualTimer
}

type manualTimer struct {
	clock *manualClock
	at    time.Time
	f     func()
	done  bool
}

func newManualClock() *manualClock {
	return &manualClock{now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
}

func (c *manualClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *manualClock) AfterFunc(d time.Duration, f func()) Timer {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := &manualTimer{clock: c, at: c.now.Add(d), f: f}
	c.timers = append(c.timers, t)
	return t
}

func (t *manualTimer) Stop() bool {
	t.clock.mu.Lock()
	defer t.clock.mu.Unlock()
	pending := !t.done
	t.done = true
	return pending
}

// Advance moves the clock forward by d, firing what falls due on the way.
func (c *manualClock) Advance(d time.Duration) {
	c.mu.Lock()
	target := c.now.Add(d)
	c.mu.Unlock()
	for {
		c.mu.Lock()
		next := c.nextDueLocked(target)
		if next == nil {
			c.now = target
			c.mu.Unlock()
			return
		}
		c.now = next.at
		next.done = true
		c.mu.Unlock()
		next.f()
	}
}

func (c *manualClock) nextDueLocked(target time.Time) *manualTimer {
	sort.SliceStable(c.timers, func(i, j int) bool { return c.timers[i].at.Before(c.timers[j].at) })
	kept := c.timers[:0]
	var due *manualTimer
	for _, t := range c.timers {
		if t.done {
			continue
		}
		kept = append(kept, t)
		if due == nil && !t.at.After(target) {
			due = t
		}
	}
	c.timers = kept
	return due
}
