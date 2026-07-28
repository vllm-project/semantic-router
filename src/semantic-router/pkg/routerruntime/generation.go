package routerruntime

import (
	"errors"
	"sync"
)

// Generation accumulates the closeable resources produced by one runtime
// build (e.g. buildRouterComponents) and tears them down in reverse
// construction order. It lets a constructor sequence register a closer
// immediately after each successful step and roll back everything built so
// far the moment a later step fails, instead of leaking partially
// constructed resources.
type Generation struct {
	mu      sync.Mutex
	closers []func() error
	closed  bool
}

// NewGeneration returns an empty Generation ready to accumulate closers.
func NewGeneration() *Generation {
	return &Generation{}
}

// Defer registers closer to run, in reverse registration order, the next
// time Close is called. Nil closers are ignored so callers can pass a
// resource's Close method directly even when the resource itself may be nil.
func (g *Generation) Defer(closer func() error) {
	if g == nil || closer == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closers = append(g.closers, closer)
}

// Close runs every registered closer exactly once, in reverse registration
// order, and joins any errors they return. It is safe to call multiple
// times or concurrently; only the first call runs the closers.
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
