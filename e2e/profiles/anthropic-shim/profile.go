package anthropicshim

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/anthropic-shim/values.yaml"

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
// shim) and points the Envoy AI Gateway routing at it so that Anthropic-
// protocol requests reach an endpoint that speaks Anthropic Messages natively.
// This is required for cache-cycle and stop-sequence assertions that exercise
// the outbound emitter's buildAnthropicUsage and stop-reason mapping paths.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new anthropic-shim profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "anthropic-shim",
			SemanticRouterValuesFile: valuesFile,
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

// Setup deploys the shared gateway stack and anthropic-shim backend.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
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
