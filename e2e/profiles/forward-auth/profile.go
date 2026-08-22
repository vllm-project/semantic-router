package forwardauth

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/forward-auth/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/forward-auth/gateway-resources/backend.yaml",
		"e2e/profiles/forward-auth/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "default", Name: "forward-be"},
		{Namespace: "default", Name: "static-be"},
	}
)

// Profile implements the forward-auth test profile.
// It exercises the forward_authorization_header feature and the internal-leg
// trust boundary: a forward-auth backend that requires the caller's inbound
// Authorization header, a static-key backend that does not, and rejection of
// spoofed reserved internal headers. Looper-leg enforcement is covered by the
// Go unit tests (see e2e/testcases/forward_authorization.go for why it is not
// asserted in-cluster here).
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new forward-auth profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "forward-auth",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "forward-auth"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests forward_authorization_header passthrough and the internal-leg trust boundary (direct forward, per-backend static-key, and spoofed reserved-header paths)"
}

// Setup deploys the shared gateway stack and forward-auth resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and forward-auth resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"forward-authorization",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
