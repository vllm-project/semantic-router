package aigateway

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/ai-gateway/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the default Kubernetes baseline test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the default Kubernetes profile backed by the shared AI Gateway stack.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "ai-gateway",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "kubernetes"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Semantic Router through the default Kubernetes baseline powered by Envoy AI Gateway"
}

// Setup deploys the shared gateway stack.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(testmatrix.BaselineRouterContract)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
