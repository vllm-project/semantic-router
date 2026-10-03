package main

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const omniDeployment = "omni-calibration"

// omniEmbedding serves the prepared bundle from a model runtime this process
// starts (VLLM_SR_RUNTIME_COMMAND, else vllm-sr-runtime on PATH) and returns
// the production embedding provider over it. stop closes the provider and the
// runtime.
func omniEmbedding(ctx context.Context, bundle string, maxTokens int) (provider *serving.EmbeddingProvider, stop func(), err error) {
	deployment := config.ModelDeployment{
		Provider: config.ModelRuntimeProvider,
		Device:   "cpu",
		Artifact: bundle,
		Input:    config.ModelInputBudget{MaxTokens: maxTokens, Overflow: "reject"},
	}.WithDefaults()
	manager := modelservice.NewManager()
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{omniDeployment: deployment})
	if err != nil {
		_ = manager.Shutdown(ctx)
		return nil, nil, err
	}
	shutdown := func() {
		_ = lease.Close()
		_ = manager.Shutdown(context.Background())
	}
	provider, err = serving.New(lease, nil).Embedding(ctx, config.ResolvedModelBinding{
		Recipe:     "image-calibration",
		Name:       "embedding",
		Binding:    config.ModelBinding{Deployment: omniDeployment, Contract: "embedding.v1"},
		Deployment: deployment,
	}, 0, 0)
	if err != nil {
		shutdown()
		return nil, nil, err
	}
	return provider, func() {
		_ = provider.Close()
		shutdown()
	}, nil
}
