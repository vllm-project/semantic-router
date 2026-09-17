package stickytoolselectionexpiry

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const stickyToolSelectionTTLSeconds = "30"

// Profile isolates the short local-store TTL from restart recovery coverage.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the local-store expiry profile.
func NewProfile() *Profile {
	stackConfig := aigateway.BaseStackConfig("sticky-tool-selection-expiry")
	stackConfig.SemanticRouterSet = map[string]string{
		"config.global.stores.tool_sessions.ttl_seconds": stickyToolSelectionTTLSeconds,
	}
	return &Profile{stack: gatewaystack.New(stackConfig)}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "sticky-tool-selection-expiry"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests local sticky tool-selection state expiry with a short isolated TTL"
}

// Setup deploys the shared gateway stack with the short local-store TTL.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return append([]string(nil), testmatrix.StickyToolSelectionExpiryContract...)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
