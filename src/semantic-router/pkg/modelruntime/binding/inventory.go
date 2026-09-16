package binding

import (
	"cmp"
	"slices"
	"sync"
)

// PreparedBinding describes a successfully warmed, still-owned task. It omits
// execution keys and remote endpoints, which may contain private connection data.
type PreparedBinding struct {
	Identity   Identity
	Artifact   string
	Capability Capability
}

// Inventory belongs to one runtime generation, independently of its shared
// resource pool. Loading or failed candidate resources never imply readiness.
type Inventory struct {
	mu       sync.RWMutex
	bindings map[*bindingLifecycle]PreparedBinding
}

func NewInventory() *Inventory {
	return &Inventory{bindings: make(map[*bindingLifecycle]PreparedBinding)}
}

func (i *Inventory) Observe(event Event) {
	if event.instance == nil || (event.State != "ready" && event.State != "closed") {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if event.State == "closed" {
		delete(i.bindings, event.instance)
		return
	}
	i.bindings[event.instance] = PreparedBinding{
		Identity: event.Identity, Artifact: event.Artifact,
		Capability: cloneCapability(event.Capability),
	}
}

// Snapshot copies metadata only; it neither prepares nor leases model resources.
func (i *Inventory) Snapshot() []PreparedBinding {
	i.mu.RLock()
	defer i.mu.RUnlock()
	result := make([]PreparedBinding, 0, len(i.bindings))
	for _, entry := range i.bindings {
		entry.Capability = cloneCapability(entry.Capability)
		result = append(result, entry)
	}
	slices.SortFunc(result, func(a, b PreparedBinding) int {
		return cmp.Or(cmp.Compare(a.Identity.Recipe, b.Identity.Recipe),
			cmp.Compare(a.Identity.Name, b.Identity.Name),
			cmp.Compare(a.Identity.Deployment, b.Identity.Deployment),
			cmp.Compare(a.Artifact, b.Artifact),
			cmp.Compare(a.Capability.Device, b.Capability.Device))
	})
	return result
}
