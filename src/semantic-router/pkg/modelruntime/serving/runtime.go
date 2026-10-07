// Package serving implements the router's typed task bindings through
// model_runtime deployments. It keeps the binding facade (Sequence, Scores,
// windows, Tokens, Grounded, OperatingPoint, Embedding, Relevance and their
// diagnostics) so consumers change only their constructor. Tokenization,
// windows and head readouts run in the runtime; this package checks each
// binding against its deployment's model card before the binding is
// published and converts results to the router's task types once.
package serving

import (
	"context"
	"errors"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/diagnostics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
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
	Classify(ctx context.Context, deployment string, request modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error)
	Embed(ctx context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error)
	Rerank(ctx context.Context, deployment string, request modelservice.RerankRequest) (modelservice.RerankResponse, error)
}

// Runtime prepares task bindings for one router generation. Pool may be
// shared with the preceding generation; each returned binding owns an
// independent reference to its deployment's admission gate.
type Runtime struct {
	Pool            *binding.Pool
	services        Services
	registry        *binding.Registry
	inventory       *binding.Inventory
	prepared        *binding.PreparedTasks
	sequence        *binding.Task[string, tasks.LabelDistribution]
	scores          *binding.Task[string, tasks.LabelScores]
	sequenceWindows *binding.Task[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution]
	scoreWindows    *binding.Task[tasks.TextWindowsRequest, tasks.WindowedLabelScores]
	tokens          *binding.Task[string, tasks.TokenClassificationResult]
	tokenWindows    *binding.Task[tasks.TextWindowsRequest, tasks.WindowedTokenClassification]
	grounded        *binding.Task[tasks.GroundedTextRequest, tasks.TokenClassificationResult]

	mu              sync.Mutex
	operatingPoints map[*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]]*OperatingPointScorer
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
	r := &Runtime{
		Pool: pool, services: services, registry: registry, inventory: inventory, prepared: prepared,
		operatingPoints: make(map[*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]]*OperatingPointScorer),
	}
	r.sequence = mustRegister(registry, config.RemoteClassifierContractLabelDistribution, config.RemoteClassifierContractLabelDistribution, validateText, validateDistributionResult)
	r.scores = mustRegister(registry, config.RemoteClassifierContractLabelScores, config.RemoteClassifierContractLabelScores, validateText, validateScoresResult)
	r.sequenceWindows = mustRegister(registry, "windowed_label_distribution.v1", config.RemoteClassifierContractLabelDistribution, validateWindowInput, validateWindowDistribution)
	r.scoreWindows = mustRegister(registry, "windowed_label_scores.v1", config.RemoteClassifierContractLabelScores, validateWindowInput, validateWindowScores)
	r.tokens = mustRegister(registry, config.RemoteClassifierContractTokenSpans, config.RemoteClassifierContractTokenSpans, validateText, validateSpans)
	r.tokenWindows = mustRegister(registry, "windowed_token_spans.v1", config.RemoteClassifierContractTokenSpans, validateWindowInput, validateWindowTokens)
	r.grounded = mustRegister(registry, "grounded_text.v1", config.RemoteClassifierContractTokenSpans, validateGroundedInput, validateGroundedSpans)
	return r
}

func mustRegister[I, O any](registry *binding.Registry, taskID, contract string, input func(I) error, output func(I, O) error) *binding.Task[I, O] {
	task, err := binding.RegisterTask(registry, taskID, contract, input, output)
	if err != nil {
		panic(err)
	}
	return task
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
	adapter := spec.Binding.Adapter
	if adapter == "" {
		adapter = "auto"
	}
	return binding.Identity{Recipe: string(spec.Recipe), Name: spec.Name, Deployment: spec.Binding.Deployment, Contract: spec.Binding.Contract, Adapter: adapter, Head: spec.Binding.Head}
}
