package vectorstoreregistry

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
)

const (
	valuesFile       = "e2e/profiles/vectorstore-registry/values.yaml"
	postgresManifest = "deploy/kubernetes/router-replay/postgres.yaml"
)

var resourceManifests = []string{
	"deploy/kubernetes/response-api/gwapi-resources.yaml",
}

// Profile implements the Vector Store Registry durability test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new Vector Store Registry profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "vectorstore-registry",
			SemanticRouterValuesFile: valuesFile,
			// provider-mocker serves the embedding fixture, which the Router
			// prepares as it starts.
			PrerequisiteManifests: []string{postgresManifest, "deploy/kubernetes/response-api/provider-mocker.yaml"},
			PrerequisiteWaits:     []helpers.DeploymentRef{{Namespace: "default", Name: "provider-mocker"}},
			ResourceManifests:     resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "vectorstore-registry"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests that vector store records in Postgres and file bytes on the Router's data volume survive a pod restart"
}

// Setup deploys Postgres, the router, and gateway resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"vectorstore-registry-restart-recovery",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
