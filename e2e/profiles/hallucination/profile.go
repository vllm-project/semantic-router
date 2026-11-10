// Package hallucination provides the e2e test profile for the hallucination
// signal on a remote systemone deployment. It deploys the router with a mock
// LLM backend whose provider-mocker also serves the System One fixture that
// answers the hallucination rule, then verifies that hallucination detection
// surfaces on the response.
package hallucination

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/hallucination/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	"deploy/kubernetes/hallucination/gwapi-resources.yaml",
}

// waitDeployments are the mock backends the profile must wait on before running
// tests: provider-mocker is both the LLM backend and the hallucination
// deployment.
var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "provider-mocker"},
}

// Profile implements the hallucination test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new hallucination profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "hallucination",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "hallucination"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the hallucination signal on a remote systemone deployment and the response_guard that acts on it"
}

// Setup deploys the shared gateway stack and hallucination resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and hallucination resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"hallucination-detection",
		"hallucination-streaming-observation",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
