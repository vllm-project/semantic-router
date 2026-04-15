package vectorstoreregistry

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
)

const (
	valuesFile       = "e2e/profiles/vectorstore-registry/values.yaml"
	postgresManifest = "deploy/kubernetes/router-replay/postgres.yaml"
)

var resourceManifests = []string{
	"deploy/kubernetes/response-api/mock-vllm.yaml",
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
			PrerequisiteManifests:    []string{postgresManifest},
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "vectorstore-registry"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Vector Store Registry metadata persistence using Postgres across pod restarts"
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
