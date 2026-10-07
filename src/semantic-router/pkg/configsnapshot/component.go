package configsnapshot

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
)

// Component is a part of the serving Router built from a snapshot.
type Component string

const (
	// ComponentRouter is the routing pipeline. Every change rebuilds it.
	ComponentRouter Component = "router"
	// ComponentSignals extracts signals: the recipe classifiers, embeddings
	// and rerankers, and the model runtime deployments they run on.
	ComponentSignals Component = "signals"
	// ComponentUpstream is the upstream layer: clusters, endpoints, their
	// connection pools and health.
	ComponentUpstream Component = "upstream"
)

// Dependencies says which resources each component is built from. A
// candidate whose resources of those kinds equal the active snapshot's keeps
// the active component, the same instance, instead of building another.
var Dependencies = map[Component][]Kind{
	ComponentRouter:   Kinds,
	ComponentSignals:  {KindRuntimeModel, KindSettings, KindProgram},
	ComponentUpstream: {KindEndpoint, KindCluster, KindListener},
}

// ComponentKey identifies what component c is built from in the snapshot:
// two snapshots with equal keys can serve with one instance of c. Keys are
// comparable within one process only.
func (s *Snapshot) ComponentKey(c Component) string {
	var inputs []any
	for _, kind := range Dependencies[c] {
		for _, resource := range s.resources.List(kind) {
			inputs = append(inputs, resource.Ref.String(), resource.Hash)
		}
	}
	return fingerprint(c, inputs)
}

// Part is a component a snapshot owns besides the routing pipeline, such as
// the upstream layer.
type Part interface {
	// Close releases the part once no snapshot serves with it. ctx bounds a
	// drain of work still in flight.
	Close(ctx context.Context) error
}

// PartBuilder builds one component for each snapshot.
type PartBuilder struct {
	Component Component
	// Validate, when set, checks the candidate against the active snapshot,
	// which is nil for the first one, without building anything.
	Validate func(candidate, active *Snapshot) error
	// Build builds the part for candidate. previous is the active snapshot's
	// part, or nil; Build may adopt pieces of it but must not close it.
	Build func(ctx context.Context, candidate *Snapshot, previous Part) (Part, error)
	// Warm, when set, returns once the part can serve.
	Warm func(ctx context.Context, part Part) error
}

// sharedPart counts the snapshots that serve with one part.
type sharedPart struct {
	part Part
	refs atomic.Int32
}

func newSharedPart(part Part) *sharedPart {
	shared := &sharedPart{part: part}
	shared.refs.Store(1)
	return shared
}

func (p *sharedPart) retain() *sharedPart {
	p.refs.Add(1)
	return p
}

func (p *sharedPart) release(ctx context.Context) error {
	if p.refs.Add(-1) == 0 {
		return p.part.Close(ctx)
	}
	return nil
}

// parts are the components a snapshot owns. They are set while the snapshot
// is a candidate and never change once it activates.
type parts struct {
	byComponent map[Component]*sharedPart
	reused      []Component
	release     sync.Once
	releaseErr  error
}

// Part returns the snapshot's part for component c, or nil.
func (s *Snapshot) Part(c Component) Part {
	if shared := s.parts.byComponent[c]; shared != nil {
		return shared.part
	}
	return nil
}

// Reused lists the components the snapshot kept from the snapshot it
// replaced, instead of building them again.
func (s *Snapshot) Reused() []Component {
	return append([]Component(nil), s.parts.reused...)
}

// Release lets go of the snapshot's parts once nothing serves with it any
// more; a part closes when the last snapshot that holds it lets go. Only the
// first call counts.
func (s *Snapshot) Release(ctx context.Context) error {
	if s == nil {
		return nil
	}
	s.parts.release.Do(func() {
		var errs []error
		for _, shared := range s.parts.byComponent {
			errs = append(errs, shared.release(ctx))
		}
		s.parts.releaseErr = errors.Join(errs...)
	})
	return s.parts.releaseErr
}

// buildParts builds, or keeps from the active snapshot, every part the
// builders make. On failure it releases what it built and kept.
func buildParts(ctx context.Context, builders []PartBuilder, candidate *Candidate) error {
	active := candidate.active
	snapshot := candidate.snapshot
	snapshot.parts.byComponent = make(map[Component]*sharedPart, len(builders))
	for _, builder := range builders {
		var previous *sharedPart
		if active != nil {
			previous = active.parts.byComponent[builder.Component]
		}
		if previous != nil && active.ComponentKey(builder.Component) == snapshot.ComponentKey(builder.Component) {
			snapshot.parts.byComponent[builder.Component] = previous.retain()
			candidate.RecordReuse(builder.Component)
			continue
		}
		var previousPart Part
		if previous != nil {
			previousPart = previous.part
		}
		part, err := builder.Build(ctx, snapshot, previousPart)
		if err != nil {
			return errors.Join(err, snapshot.Release(context.WithoutCancel(ctx)))
		}
		snapshot.parts.byComponent[builder.Component] = newSharedPart(part)
		if builder.Warm != nil {
			if err := builder.Warm(ctx, part); err != nil {
				if ctx.Err() == nil {
					err = Reject(StageWarm, CodeWarmupFailed, err)
				}
				return errors.Join(err, snapshot.Release(context.WithoutCancel(ctx)))
			}
		}
	}
	return nil
}
