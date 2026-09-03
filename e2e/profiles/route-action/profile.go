// Package routeaction provides the e2e test profile for #3175's decision
// route action. It deploys the router with prompt_guard pointed at a
// deliberately unreachable endpoint and on_error: block, so the jailbreak
// signal fires for every request; a keyword marker separates detected prompt
// attacks from benign traffic. The tests verify that a detected request is
// routed to the action destination and a benign request keeps its normal
// route.
package routeaction

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile           = "e2e/profiles/route-action/values.yaml"
	mappingConfigMapYAML = "deploy/kubernetes/route-action/jailbreak-mapping-configmap.yaml"
)

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/mock-vllm.yaml",
	"deploy/kubernetes/route-action/gwapi-resources.yaml",
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "mock-vllm"},
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
			PrerequisiteManifests:    []string{mappingConfigMapYAML},
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
	return "Tests the decision route action end-to-end for detected and benign requests"
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
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
