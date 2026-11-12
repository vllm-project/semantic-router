// Package routeaction provides the e2e test profile for #3175's decision
// route action. It deploys the router with its jailbreak rule answered by
// provider-mocker's System One fixture, which detects the attack phrase the
// attack prompt carries. The tests verify that a detected request is routed
// to the action destination and a benign request keeps its normal route.
package routeaction

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/route-action/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	"deploy/kubernetes/route-action/gwapi-resources.yaml",
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "provider-mocker"},
}

// Profile implements the decision route action test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new route-action profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "route-action",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "route-action"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the decision route action end-to-end for detected and benign requests, and response_guard on jailbreak content in model output"
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
		"route-action-detected",
		"route-action-benign",
		// The response-guard recipe, formerly its own profile.
		"response-jailbreak-window-block",
		"response-jailbreak-window-warning",
		"response-jailbreak-streaming-passthrough",
		"response-jailbreak-streaming-block",
		"response-jailbreak-streaming-pass",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
