package multiendpoint

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/multi-endpoint/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/multi-endpoint/gateway-resources/backend.yaml",
		"e2e/profiles/multi-endpoint/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "default", Name: "vllm-14b-dev"},
		{Namespace: "default", Name: "vllm-14b-prod"},
	}
)

// Profile implements the multi-endpoint test profile.
// It demonstrates environment-based routing (Dev vs Prod) with
// different PII and jailbreak policies per environment, using
// model aliases and external_model_ids for model name rewriting.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new multi-endpoint profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "multi-endpoint",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "multi-endpoint"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests multi-endpoint routing with per-environment PII & jailbreak safety policies (Dev vs Prod)"
}

// Setup deploys the shared gateway stack and multi-endpoint resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and multi-endpoint resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"pii-detection",
		"jailbreak-detection",
		"chat-completions-request",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
