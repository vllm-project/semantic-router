package serving

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Head kinds and input forms a classify head card declares.
const (
	kindSequence  = "sequence"
	kindScores    = "scores"
	kindToken     = "token"
	inputText     = "text"
	inputGrounded = "grounded"
)

// target is a binding resolved against its deployment's card: the head the
// binding runs and the resource reference that carries its admission gate.
type target struct {
	spec       config.ResolvedModelBinding
	deployment string
	card       modelservice.ModelCard
	head       modelservice.HeadCard
	resource   *binding.Resource
	// scan is a question deployment's scan budget (0: the model's own).
	scan int
}

// deploymentPlanner is implemented by services that can start a deployment
// their generation's plan did not include (modelservice.Lease).
type deploymentPlanner interface {
	Ensure(name string, deployment config.ModelDeployment) error
}

// prepareHead resolves a classify binding: the deployment's card must serve
// classify with a head of the wanted kind that accepts the input form.
func (r *Runtime) prepareHead(ctx context.Context, spec config.ResolvedModelBinding, kind, input string) (*target, binding.Capability, error) {
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, binding.Capability{}, err
	}
	if !card.Serves("classify") {
		return nil, binding.Capability{}, fmt.Errorf("%w: deployment %q does not serve classify", binding.ErrCapability, spec.Binding.Deployment)
	}
	head, ok := card.Head(spec.Binding.Head)
	if !ok {
		return nil, binding.Capability{}, fmt.Errorf("%w: deployment %q has no head %q", binding.ErrCapability, spec.Binding.Deployment, spec.Binding.Head)
	}
	if head.Kind != kind || !head.Accepts(input) {
		return nil, binding.Capability{}, fmt.Errorf("%w: head %q of deployment %q is a %s head, not a %s head over %s input", binding.ErrCapability, head.Name, spec.Binding.Deployment, head.Kind, kind, input)
	}
	if len(head.Labels) == 0 {
		return nil, binding.Capability{}, fmt.Errorf("%w: head %q declares no labels", binding.ErrCapability, head.Name)
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, binding.Capability{}, err
	}
	capability := binding.Capability{
		Contract: spec.Binding.Contract, Provider: Provider, Device: card.Device, Precision: card.Dtype,
		Labels: slices.Clone(head.Labels),
		Limits: binding.Limits{ModelTokens: card.MaxInputTokens, DeploymentTokens: spec.Deployment.Input.MaxTokens, Overflow: spec.Deployment.Input.Overflow},
	}
	return &target{spec: spec, deployment: spec.Binding.Deployment, card: card, head: head, resource: resource}, capability, nil
}

// Labels returns the label vocabulary, in output order, of the classify head
// the binding runs; consumers without a mapping file take their labels from
// the served model. A binding that asks a decision model's ready-made span
// question has none: its spans name their own labels. A binding that asks a
// Vela 2.0 model its signal's question has the question's options.
func (r *Runtime) Labels(ctx context.Context, spec config.ResolvedModelBinding) ([]string, error) {
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if _, ok := spanPreset(spec, card); ok {
		return nil, nil
	}
	if question, ok := questionFor(spec, card); ok {
		return question.labels(), nil
	}
	head, ok := card.Head(spec.Binding.Head)
	if !ok || len(head.Labels) == 0 {
		return nil, fmt.Errorf("%w: deployment %q has no labeled head %q", binding.ErrCapability, spec.Binding.Deployment, spec.Binding.Head)
	}
	return slices.Clone(head.Labels), nil
}

// DeploymentCard waits until a deployment is ready and returns its card, so
// preparation can check what its consumers ask of the model.
func (r *Runtime) DeploymentCard(ctx context.Context, name string, deployment config.ModelDeployment) (modelservice.ModelCard, error) {
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	return r.card(ctx, config.ResolvedModelBinding{Name: name, Binding: config.ModelBinding{Deployment: name}, Deployment: deployment.WithDefaults()})
}

// CurrentDeploymentCard reads this generation's observed ready metadata.
// Attached services may recover independently of Router startup; consumers
// can check their capabilities without waiting or discovering at request time.
func (r *Runtime) CurrentDeploymentCard(name string) (modelservice.ModelCard, bool) {
	if r.services == nil {
		return modelservice.ModelCard{}, false
	}
	return r.services.CurrentCard(name)
}

// preparationContext bounds a preparation whose caller set no deadline, so a
// runtime that never becomes ready fails the generation instead of blocking it.
func preparationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, modelservice.ReadyTimeout())
}

// card waits for the deployment's card.
func (r *Runtime) card(ctx context.Context, spec config.ResolvedModelBinding) (modelservice.ModelCard, error) {
	if !spec.Deployment.IsModelRuntime() {
		return modelservice.ModelCard{}, fmt.Errorf("%w: provider %q is not served by the model runtime", binding.ErrCapability, spec.Deployment.Provider)
	}
	if r.services == nil {
		return modelservice.ModelCard{}, ErrNotConfigured
	}
	if planner, ok := r.services.(deploymentPlanner); ok {
		if err := planner.Ensure(spec.Binding.Deployment, spec.Deployment); err != nil {
			return modelservice.ModelCard{}, fmt.Errorf("model_runtime deployment %q: %w", spec.Binding.Deployment, err)
		}
	}
	card, err := r.services.Card(ctx, spec.Binding.Deployment)
	if err != nil {
		return modelservice.ModelCard{}, fmt.Errorf("model_runtime deployment %q: %w", spec.Binding.Deployment, err)
	}
	return card, nil
}

// acquire takes the deployment's admission gate; the runtime process owns
// the model. Bindings of one deployment share one gate, keyed by the served
// model's identity so a changed model never inherits the gate of the model it
// replaced.
func (r *Runtime) acquire(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard) (*binding.Resource, error) {
	if card.Device == "" || card.Dtype == "" {
		return nil, fmt.Errorf("%w: the card of deployment %q reports no device or dtype", binding.ErrCapability, spec.Binding.Deployment)
	}
	execution, _ := json.Marshal(struct {
		Deployment, Model, Profile string
	}{spec.Binding.Deployment, card.ID, card.Profile})
	identity := binding.ResourceIdentity{
		Artifact: card.ModelSHA256, Revision: card.Revision, Provider: Provider,
		Device: card.Device, Precision: card.Dtype, Execution: string(execution),
	}
	if identity.Artifact == "" {
		identity.Artifact = card.ID
	}
	budget, gate := resourceAdmission(spec)
	return r.Pool.Admit(ctx, identity, budget, gate)
}

func resourceAdmission(spec config.ResolvedModelBinding) (string, admission.Admissioner) {
	data, _ := json.Marshal(spec.Admission)
	if spec.Admission.MaxConcurrency == 0 {
		return string(data), admission.Noop{}
	}
	return string(data), admission.NewSemaphore(spec.Admission.MaxConcurrency, spec.Admission.MaxQueue, time.Duration(spec.Admission.QueueTimeoutMs)*time.Millisecond, admission.Overflow(spec.Admission.OnOverflow))
}

// publish resolves a prepared task, runs one warmup call through the whole
// path (request options, transport, decoding, validation) and marks it ready.
// A failure releases the target's resource reference.
func publish[I, O any](ctx context.Context, task *binding.Task[I, O], t *target, capability binding.Capability, infer func(context.Context, io.Closer, I) (O, error), warmup I) (*binding.Resolved[I, O], error) {
	bound, err := task.Resolve(taskIdentity(t.spec), capability, t.resource, infer)
	if err == nil {
		_, err = bound.Call(ctx, string(t.spec.Recipe), warmup)
	}
	if err != nil {
		_ = t.resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}
