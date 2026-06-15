package responseapi

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/response-api/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/response-api/mock-vllm.yaml",
	"deploy/kubernetes/response-api/gwapi-resources.yaml",
}

// Profile implements the Response API test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new Response API profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "response-api",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "response-api"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Response API endpoints (POST/GET/DELETE /v1/responses)"
}

// Setup deploys the shared gateway stack and Response API resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and Response API resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
		"response-api-conversation-chaining",
		"response-api-error-missing-input",
		"response-api-error-nonexistent-previous-response-id",
		"response-api-error-nonexistent-response-id-get",
		"response-api-error-nonexistent-response-id-delete",
		"response-api-error-backend-passthrough",
		"response-api-edge-empty-input",
		"response-api-edge-large-input",
		"response-api-edge-special-characters",
		"response-api-edge-concurrent-requests",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
