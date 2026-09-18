package anthropicshim

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile                 = "e2e/profiles/anthropic-shim/values.yaml"
	mockEmbeddingManifest      = "e2e/profiles/remote-embedding/manifests/mock-embedding.yaml"
	mockEmbeddingNamespace     = "default"
	mockEmbeddingDeployment    = "mock-embedding"
	mockEmbeddingReadyTimeout  = 5 * time.Minute
	mockEmbeddingReadyInterval = 2 * time.Second
)

var (
	resourceManifests = []string{
		"e2e/profiles/anthropic-shim/gateway-resources/backend.yaml",
		"e2e/profiles/anthropic-shim/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "anthropic-backend-system", Name: "anthropic-backend-qwen"},
	}
)

// Profile implements the anthropic-shim test profile.
//
// It deploys the anthropic-shim backend (llama.cpp + the Python translation
// shim) behind a plain Gateway API route. The semantic Router is the only
// protocol translator in this profile, so failures identify codec regressions
// instead of interactions with a second protocol processor.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new anthropic-shim profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "anthropic-shim",
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    []string{mockEmbeddingManifest},
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "anthropic-shim"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Anthropic /v1/messages response shape and cache-cycle behaviour against the llama.cpp anthropic-shim backend"
}

// Setup deploys the embedding fixture before the Router because startup probes
// call the configured provider immediately.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.stack.ApplyPrerequisites(ctx, opts); err != nil {
		return err
	}
	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required to verify the mock embedding provider")
	}
	if err := helpers.WaitForDeploymentReady(
		ctx,
		opts.KubeClient,
		mockEmbeddingNamespace,
		mockEmbeddingDeployment,
		mockEmbeddingReadyTimeout,
		mockEmbeddingReadyInterval,
		opts.Verbose,
	); err != nil {
		return fmt.Errorf("wait for mock embedding provider: %w", err)
	}
	if err := p.stack.DeployCore(ctx, opts); err != nil {
		return err
	}
	if err := p.stack.ApplyResources(ctx, opts); err != nil {
		return err
	}
	return p.stack.Verify(ctx, opts)
}

// Teardown removes the shared gateway stack and anthropic-shim backend.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.AnthropicShimContract
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
