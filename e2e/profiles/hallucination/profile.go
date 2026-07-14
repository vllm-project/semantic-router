// Package hallucination provides the e2e test profile for the pluggable
// hallucination detector endpoint backend. It deploys the router with the
// endpoint detector backend, a mock LLM backend, and a mock detector endpoint,
// then verifies that hallucination detection surfaces on the response.
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
	"deploy/kubernetes/hallucination/mock-vllm.yaml",
	"deploy/kubernetes/hallucination/mock-hallucination-detector.yaml",
	"deploy/kubernetes/hallucination/gwapi-resources.yaml",
}

// waitDeployments are the mock backends the profile must wait on before running
// tests: the LLM backend and the endpoint detector both back the detection path.
var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "mock-vllm"},
	{Namespace: "default", Name: "mock-hallucination-detector"},
}

// Profile implements the hallucination endpoint-backend test profile.
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
	return "Tests the pluggable hallucination detector endpoint backend end-to-end"
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
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}

