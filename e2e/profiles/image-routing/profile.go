// Package imagerouting provides the e2e test profile for routing on what a
// request's images show and for decisions that answer with a generated image.
// provider-mocker's System One fixture answers the image question by the
// marker an image carries, and the modality question by the marker a prompt
// carries; provider-mocker's images and Responses endpoints generate the image.
package imagerouting

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/image-routing/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	// Every path goes through the Router's extproc to provider-mocker.
	"deploy/kubernetes/route-action/gwapi-resources.yaml",
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "provider-mocker"},
}

// Profile implements the image routing test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new image-routing profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "image-routing",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "image-routing"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests routing on a request's images through System One and decisions that answer with a generated image"
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
		"systemone-image-input-routing",
		"output-contract-image-generation",
		"output-contract-text-and-image",
		"images-api-entry",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
