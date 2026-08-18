// Package jailbreakonerror provides the e2e test profile for #2918's
// PromptGuardConfig.OnError contract. It deploys the router with
// prompt_guard.protocol pointed at a deliberately unreachable endpoint and
// on_error: block, then verifies that a classify failure closes the request
// instead of silently letting it through.
package jailbreakonerror

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/jailbreak-onerror/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/mock-vllm.yaml",
	"deploy/kubernetes/jailbreak-onerror/gwapi-resources.yaml",
}

// waitDeployments are the mock backends the profile must wait on before
// running tests. mock-vllm exists only to satisfy config validation (a
// modelRef needs a real backend) - the request path under test never
// reaches it, since on_error: block short-circuits before model selection.
var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "mock-vllm"},
}

// Profile implements the PromptGuardConfig.OnError test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new jailbreak-onerror profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "jailbreak-onerror",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "jailbreak-onerror"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests PromptGuardConfig.OnError: block against a real classify failure end-to-end"
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
	return []string{
		"jailbreak-onerror-block",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
