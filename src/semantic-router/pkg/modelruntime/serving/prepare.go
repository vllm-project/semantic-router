package serving

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// deploymentPlanner is implemented by services that can start a deployment
// their generation's plan did not include (modelservice.Lease).
type deploymentPlanner interface {
	Ensure(name string, deployment config.ModelDeployment) error
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

// card waits for the deployment's card: a model_runtime deployment's, or
// the one a systemone deployment's API lists (see WithRemote).
func (r *Runtime) card(ctx context.Context, spec config.ResolvedModelBinding) (modelservice.ModelCard, error) {
	if !spec.Deployment.IsModelRuntime() && !spec.Deployment.IsSystemOne() {
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
		Deployment, Model string
		EngineArgs        modelservice.EngineArgs
	}{spec.Binding.Deployment, card.ID, card.EngineArgs})
	identity := binding.ResourceIdentity{
		Artifact: card.ModelSHA256, Revision: card.Revision, Provider: providerOf(spec),
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
