package responseapishared

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
)

// ValkeyProfile implements the shared setup for Valkey-backed profiles.
type ValkeyProfile struct {
	name        string
	description string
	stack       *gatewaystack.Stack
}

// NewValkeyProfile constructs a shared Valkey-backed profile.
func NewValkeyProfile(name, description, valuesFile, valkeyManifest string) *ValkeyProfile {
	return &ValkeyProfile{
		name:        name,
		description: description,
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     name,
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    []string{valkeyManifest},
			ResourceManifests:        sharedResourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *ValkeyProfile) Name() string {
	return p.name
}

// Description returns the profile description.
func (p *ValkeyProfile) Description() string {
	return p.description
}

// Setup deploys the shared gateway stack and Valkey prerequisite.
func (p *ValkeyProfile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and Valkey prerequisite.
func (p *ValkeyProfile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *ValkeyProfile) GetTestCases() []string {
	return []string{
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
		"response-api-conversation-chaining",
		"response-api-edge-empty-input",
		"response-api-edge-large-input",
		"response-api-edge-special-characters",
		"response-api-edge-concurrent-requests",
		"response-api-ttl-expiry",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *ValkeyProfile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
