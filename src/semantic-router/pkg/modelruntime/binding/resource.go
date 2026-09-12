package binding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

// Pool shares provider-declared compatible physical resources across bindings
// and generations. It is owned by the router service, not a native role slot.
type Pool struct {
	mu      sync.Mutex
	entries map[string]*resourceEntry
}

type resourceEntry struct {
	ready    chan struct{}
	resource io.Closer
	gate     admission.Admissioner
	budget   string
	err      error
	refs     int
}

func NewPool() *Pool { return &Pool{entries: make(map[string]*resourceEntry)} }

// Acquire loads at most one resource for an identity. budget is a canonical
// description of its admission settings: aliases cannot create separate gates
// or silently change the total capacity of a shared physical resource.
func (p *Pool) Acquire(ctx context.Context, identity ResourceIdentity, budget string, gate admission.Admissioner, load func(context.Context) (io.Closer, error)) (*Resource, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	key, err := identity.Key()
	if err != nil {
		return nil, err
	}
	if load == nil {
		return nil, fmt.Errorf("model resource loader is required")
	}
	p.mu.Lock()
	if p.entries == nil {
		p.entries = make(map[string]*resourceEntry)
	}
	if entry, exists := p.entries[key]; exists {
		if entry.budget != budget {
			p.mu.Unlock()
			return nil, fmt.Errorf("%w: shared resource admission budgets differ", ErrCapability)
		}
		entry.refs++
		p.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), p.release(key, entry))
		case <-entry.ready:
		}
		if entry.err != nil {
			return nil, errors.Join(entry.err, p.release(key, entry))
		}
		return &Resource{pool: p, key: key, entry: entry}, nil
	}
	if gate == nil {
		gate = admission.Noop{}
	}
	entry := &resourceEntry{ready: make(chan struct{}), gate: gate, budget: budget, refs: 1}
	p.entries[key] = entry
	p.mu.Unlock()
	resource, loadErr := load(ctx)
	if loadErr == nil && resource == nil {
		loadErr = fmt.Errorf("model loader returned no resource")
	}
	if loadErr == nil {
		loadErr = ctx.Err()
	}
	if loadErr != nil && resource != nil {
		loadErr = errors.Join(loadErr, resource.Close())
		resource = nil
	}
	p.mu.Lock()
	entry.resource, entry.err = resource, loadErr
	close(entry.ready)
	p.mu.Unlock()
	if loadErr != nil {
		return nil, errors.Join(loadErr, p.release(key, entry))
	}
	return &Resource{pool: p, key: key, entry: entry}, nil
}

func (p *Pool) release(key string, entry *resourceEntry) error {
	p.mu.Lock()
	entry.refs--
	if entry.refs != 0 {
		p.mu.Unlock()
		return nil
	}
	delete(p.entries, key)
	resource := entry.resource
	p.mu.Unlock()
	if resource != nil {
		return resource.Close()
	}
	return nil
}

// Resource is one binding's ownership reference. Use holds its read lock
// across admission and execution; Close cannot unload a non-preemptible call
// even if its caller has canceled the request or retired the generation.
type Resource struct {
	mu       sync.RWMutex
	pool     *Pool
	key      string
	entry    *resourceEntry
	closed   bool
	closeErr error
	closers  []io.Closer
}

// Own attaches a task-specific adapter/head to this reference. Its lifetime
// ends before the shared physical resource is released, after all Use calls.
func (r *Resource) Own(closer io.Closer) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return ErrClosed
	}
	if closer != nil {
		r.closers = append(r.closers, closer)
	}
	return nil
}

func (r *Resource) Use(ctx context.Context, call func(io.Closer) error) error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ticket, err := r.entry.gate.Acquire(ctx)
	if err != nil {
		return err
	}
	defer ticket()
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := call(r.entry.resource); err != nil {
		return err
	}
	return ctx.Err()
}

func (r *Resource) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed {
		r.closed = true
		var closeErrors []error
		for i := len(r.closers) - 1; i >= 0; i-- {
			closeErrors = append(closeErrors, r.closers[i].Close())
		}
		closeErrors = append(closeErrors, r.pool.release(r.key, r.entry))
		r.closeErr = errors.Join(closeErrors...)
	}
	return r.closeErr
}
