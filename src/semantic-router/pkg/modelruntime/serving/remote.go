package serving

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A systemone deployment is a decision model behind a remote System One API.
// It answers a binding's questions and presets as a model_runtime decision
// model does, through the generation's remote client: its card is the one the
// API lists, read while the generation prepares, and in a request stage its
// questions travel in one call with the stage's other questions to it (see
// modelservice.Bundle). It serves no classify head, embeddings or rerank.

// providerOf is the provider a binding's capability and resource report.
func providerOf(spec config.ResolvedModelBinding) string {
	if spec.Deployment.IsSystemOne() {
		return config.SystemOneProvider
	}
	return Provider
}

// Remote is the client of a generation's systemone deployments.
type Remote interface {
	modelservice.Remote
	// Serves reports whether name is one of its deployments.
	Serves(name string) bool
}

// WithRemote returns services that serve remote's deployments through remote
// and every other deployment through services, which may be nil.
func WithRemote(services Services, remote Remote) Services {
	if remote == nil {
		return services
	}
	return remoteServices{inner: services, remote: remote}
}

type remoteServices struct {
	inner  Services
	remote Remote
}

func (s remoteServices) Card(ctx context.Context, deployment string) (modelservice.ModelCard, error) {
	if s.remote.Serves(deployment) {
		return s.remote.PrepareCard(ctx, deployment)
	}
	if s.inner == nil {
		return modelservice.ModelCard{}, ErrNotConfigured
	}
	return s.inner.Card(ctx, deployment)
}

func (s remoteServices) CurrentCard(deployment string) (modelservice.ModelCard, bool) {
	if s.remote.Serves(deployment) {
		return s.remote.Card(deployment)
	}
	if s.inner == nil {
		return modelservice.ModelCard{}, false
	}
	return s.inner.CurrentCard(deployment)
}

func (s remoteServices) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	if s.remote.Serves(deployment) {
		return s.remote.Decide(ctx, deployment, request)
	}
	decider, ok := s.inner.(modelservice.Decider)
	if !ok {
		return modelservice.Response{}, ErrNotConfigured
	}
	return decider.Decide(ctx, deployment, request)
}

func (s remoteServices) Classify(ctx context.Context, deployment string, request modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	if err := s.local(deployment, "classify"); err != nil {
		return modelservice.ClassifyResponse{}, err
	}
	return s.inner.Classify(ctx, deployment, request)
}

func (s remoteServices) Embed(ctx context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	if err := s.local(deployment, "embeddings"); err != nil {
		return modelservice.EmbedResponse{}, err
	}
	return s.inner.Embed(ctx, deployment, request)
}

func (s remoteServices) Ensure(name string, deployment config.ModelDeployment) error {
	if s.remote.Serves(name) {
		return nil
	}
	if planner, ok := s.inner.(deploymentPlanner); ok {
		return planner.Ensure(name, deployment)
	}
	return nil
}

// local refuses a surface only the model runtime serves for a systemone
// deployment, and every call when there is no model runtime.
func (s remoteServices) local(deployment, surface string) error {
	if s.remote.Serves(deployment) {
		return fmt.Errorf("%w: systemone deployment %q answers decisions, not %s", binding.ErrCapability, deployment, surface)
	}
	if s.inner == nil {
		return ErrNotConfigured
	}
	return nil
}
