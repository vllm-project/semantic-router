package routerruntime

import (
	"errors"
	"sync"
)

// Generation accumulates the closeable resources produced by one runtime build
// and tears them down in reverse construction order. A constructor registers a
// closer after each successful step, so a later failure can roll back everything
// built so far rather than leak it.
type Generation struct {
	mu      sync.Mutex
	closers []func() error
	closed  bool
}

// NewGeneration returns an empty Generation ready to accumulate closers.
func NewGeneration() *Generation {
	return &Generation{}
}

// Defer registers closer to run, in reverse registration order, the next time
// Close is called. Nil closers are ignored.
func (g *Generation) Defer(closer func() error) {
	if g == nil || closer == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closers = append(g.closers, closer)
}

// Close runs every registered closer exactly once, in reverse registration
// order, joining their errors. Safe to call repeatedly or concurrently; only the
// first call runs the closers.
func (g *Generation) Close() error {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		return nil
	}
	g.closed = true
	closers := g.closers
	g.closers = nil
	g.mu.Unlock()

	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if err := closers[i](); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
