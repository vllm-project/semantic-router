// Package serving implements the router's typed task bindings through
// model_runtime deployments: the embedding bindings recall embeds with, and
// their diagnostics. It checks each binding against its deployment's model
// card before the binding is published and converts results to the router's
// task types once.
package serving

import (
	"context"
	"errors"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/diagnostics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Provider is the effective provider of every binding this package prepares.
const Provider = config.ModelRuntimeProvider

// ErrNotConfigured means the process has no model runtime services, so no
// model_runtime binding can be prepared.
var ErrNotConfigured = errors.New("model runtime services are not configured")

// Services is one router generation's view of its model_runtime deployments.
// Calls made with a context that carries a request bundle are bundled.
type Services interface {
	// Card waits until the deployment is ready and returns its model card.
	Card(ctx context.Context, deployment string) (modelservice.ModelCard, error)
	// CurrentCard reads observed ready metadata without discovery or waiting.
	CurrentCard(deployment string) (modelservice.ModelCard, bool)
	Embed(ctx context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error)
}

// Runtime prepares task bindings for one router generation. Pool may be
// shared with the preceding generation; each returned binding owns an
// independent reference to its deployment's admission gate.
type Runtime struct {
	Pool      *binding.Pool
	services  Services
	registry  *binding.Registry
	inventory *binding.Inventory
	prepared  *binding.PreparedTasks

	mu sync.Mutex
}

// New returns a runtime over services (nil when the process has none).
func New(services Services, pool *binding.Pool) *Runtime {
	if pool == nil {
		pool = binding.NewPool()
	}
	inventory := binding.NewInventory()
	prepared := binding.NewPreparedTasks()
	registry := binding.NewRegistry(func(event binding.Event) {
		inventory.Observe(event)
		prepared.Observe(event)
		diagnostics.Observe(event)
	})
	return &Runtime{Pool: pool, services: services, registry: registry, inventory: inventory, prepared: prepared}
}

// Services returns the generation's deployment services (nil when absent).
func (r *Runtime) Services() Services { return r.services }

// Close releases nothing itself: every prepared binding owns its reference.
func (r *Runtime) Close() error { return nil }

// ObserveBinding admits typed external connectors to this generation's
// inventory without exposing their endpoint or resource compatibility key.
func (r *Runtime) ObserveBinding(event binding.Event) {
	r.inventory.Observe(event)
	r.prepared.Observe(event)
	diagnostics.Observe(event)
}

// PreparedBindings lists the ready bindings of this generation.
func (r *Runtime) PreparedBindings() []binding.PreparedBinding {
	return r.inventory.Snapshot()
}

// Failed preparation has no effective device facts to advertise. Keep the
// binding identity and error class without logging input or error bodies.
func observePreparationFailure(spec config.ResolvedModelBinding, err error) {
	if err != nil {
		diagnostics.Observe(binding.Event{Identity: taskIdentity(spec), State: "failed", Error: err})
	}
}

func taskIdentity(spec config.ResolvedModelBinding) binding.Identity {
	return binding.Identity{Recipe: string(spec.Recipe), Name: spec.Name, Deployment: spec.Binding.Deployment, Contract: spec.Contract, Adapter: "auto"}
}
