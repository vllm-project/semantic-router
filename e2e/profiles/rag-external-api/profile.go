package ragexternalapi

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/rag-external-api/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/rag-external-api/manifests/external-rag-mock.yaml",
}

// Profile exercises the external API RAG boundary through the deployed router.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the external API RAG profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "rag-external-api",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "external-rag-mock"},
				{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
			},
		}),
	}
}

func (p *Profile) Name() string {
	return "rag-external-api"
}

func (p *Profile) Description() string {
	return "Tests typed external RAG requests and exact successful-response limits"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{"rag-external-api"}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
