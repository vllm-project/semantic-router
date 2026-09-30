package binding

import (
	"errors"
	"fmt"
	"sync"
)

var ErrNotPrepared = errors.New("model binding is not prepared")

// PreparedTasks indexes owned, warmed handles for diagnostics. It never loads
// models or acquires another resource reference. Its caller must hold the owning
// generation's lease for the complete lookup and call; close removes the handle.
// Type erasure is confined to lookup, just as it is for task definitions.
type PreparedTasks struct {
	mu      sync.RWMutex
	entries map[*bindingLifecycle]preparedTask
}
type preparedTask struct {
	metadata PreparedBinding
	handle   any
}

func NewPreparedTasks() *PreparedTasks {
	return &PreparedTasks{entries: make(map[*bindingLifecycle]preparedTask)}
}

func (p *PreparedTasks) Observe(event Event) {
	if event.instance == nil || (event.State != "ready" && event.State != "closed") {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if event.State == "closed" {
		delete(p.entries, event.instance)
		return
	}
	p.entries[event.instance] = preparedTask{metadata: PreparedBinding{Identity: event.Identity, Artifact: event.Artifact, Revision: event.Revision, Capability: cloneCapability(event.Capability)}, handle: event.prepared}
}

// LookupPrepared selects exactly one ready binding in one recipe. A shared physical
// resource does not make a foreign recipe's consumer visible here.
func LookupPrepared[I, O any](p *PreparedTasks, recipe, name, contract string) (*Resolved[I, O], PreparedBinding, error) {
	var metadata PreparedBinding
	if p == nil {
		return nil, metadata, ErrNotPrepared
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	var selected *Resolved[I, O]
	for _, entry := range p.entries {
		if entry.metadata.Identity.Recipe != recipe || entry.metadata.Identity.Name != name || entry.metadata.Identity.Contract != contract {
			continue
		}
		handle, ok := entry.handle.(*Resolved[I, O])
		if !ok {
			continue
		}
		if selected != nil {
			return nil, metadata, fmt.Errorf("%w: binding has multiple prepared representations", ErrCapability)
		}
		selected, metadata = handle, entry.metadata
	}
	if selected == nil {
		return nil, metadata, ErrNotPrepared
	}
	metadata.Capability = cloneCapability(metadata.Capability)
	return selected, metadata, nil
}
