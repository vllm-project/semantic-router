package authzrbac

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/authz-rbac/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/authz-rbac/gateway-resources/backend.yaml",
		"e2e/profiles/authz-rbac/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "default", Name: "vllm-14b"},
		{Namespace: "default", Name: "vllm-7b"},
	}
)

// Profile implements the authz-rbac test profile.
// It demonstrates user-level RBAC model routing where different users
// (admin, premium, free) are routed to different models (14B vs 7B)
// based on identity headers. Auth is simulated by injecting headers
// directly in test requests.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new authz-rbac profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "authz-rbac",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "authz-rbac"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests RBAC-based model routing with simulated user identity headers (admin→14B, premium→14B/7B, free→7B)"
}

// Setup deploys the shared gateway stack and authz-specific resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and authz-specific resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"chat-completions-request-authz",
		"ratelimit-limitor",
		"authz-rbac-routing",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
