// Package localclassifierbackend validates the classifier signal's local
// (in-process Candle) backend end-to-end, isolated from the shared
// ai-gateway profile's safety-decision priority space (block_jailbreak,
// block_pii). See #3756/#3178.
package localclassifierbackend

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/local-classifier-backend/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the classifier signal's local backend in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new local-classifier-backend profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "local-classifier-backend",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "local-classifier-backend" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the classifier signal's local (in-process Candle) backend end-to-end"
}

// Setup deploys the shared gateway stack and this profile's resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and this profile's resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{"local-classifier-routing"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
