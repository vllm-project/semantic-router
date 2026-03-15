package all

import (
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	aibrix "github.com/vllm-project/semantic-router/e2e/profiles/aibrix"
	authzrbac "github.com/vllm-project/semantic-router/e2e/profiles/authz-rbac"
	dashboard "github.com/vllm-project/semantic-router/e2e/profiles/dashboard"
	dynamicconfig "github.com/vllm-project/semantic-router/e2e/profiles/dynamic-config"
	dynamo "github.com/vllm-project/semantic-router/e2e/profiles/dynamo"
	istio "github.com/vllm-project/semantic-router/e2e/profiles/istio"
	llmd "github.com/vllm-project/semantic-router/e2e/profiles/llm-d"
	mlmodelselection "github.com/vllm-project/semantic-router/e2e/profiles/ml-model-selection"
	multiendpoint "github.com/vllm-project/semantic-router/e2e/profiles/multi-endpoint"
	productionstack "github.com/vllm-project/semantic-router/e2e/profiles/production-stack"
	raghybridsearch "github.com/vllm-project/semantic-router/e2e/profiles/rag-hybrid-search"
	responseapi "github.com/vllm-project/semantic-router/e2e/profiles/response-api"
	responseapiredis "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis"
	responseapirediscluster "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis-cluster"
	routingstrategies "github.com/vllm-project/semantic-router/e2e/profiles/routing-strategies"
	streaming "github.com/vllm-project/semantic-router/e2e/profiles/streaming"
)

var mockVLLMLocalImages = []framework.LocalImageBuild{
	{
		Dockerfile:   "tools/mock-vllm/Dockerfile",
		Tag:          "ghcr.io/vllm-project/semantic-router/mock-vllm:latest",
		BuildContext: "tools/mock-vllm",
	},
}

func init() {
	register("ai-gateway", func() framework.Profile { return aigateway.NewProfile() }, framework.ProfileCapabilities{})
	register("aibrix", func() framework.Profile { return aibrix.NewProfile() }, framework.ProfileCapabilities{})
	register("authz-rbac", func() framework.Profile { return authzrbac.NewProfile() }, framework.ProfileCapabilities{})
	register("dashboard", func() framework.Profile { return dashboard.NewProfile() }, framework.ProfileCapabilities{})
	register("dynamic-config", func() framework.Profile { return dynamicconfig.NewProfile() }, framework.ProfileCapabilities{})
	register("dynamo", func() framework.Profile { return dynamo.NewProfile() }, framework.ProfileCapabilities{RequiresGPU: true})
	register("istio", func() framework.Profile { return istio.NewProfile() }, framework.ProfileCapabilities{})
	register("llm-d", func() framework.Profile { return llmd.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"ml-model-selection",
		func() framework.Profile { return mlmodelselection.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register("multi-endpoint", func() framework.Profile { return multiendpoint.NewProfile() }, framework.ProfileCapabilities{})
	register("production-stack", func() framework.Profile { return productionstack.NewProfile() }, framework.ProfileCapabilities{})
	register("rag-hybrid-search", func() framework.Profile { return raghybridsearch.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"response-api",
		func() framework.Profile { return responseapi.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register(
		"response-api-redis",
		func() framework.Profile { return responseapiredis.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register(
		"response-api-redis-cluster",
		func() framework.Profile { return responseapirediscluster.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register("routing-strategies", func() framework.Profile { return routingstrategies.NewProfile() }, framework.ProfileCapabilities{})
	register("streaming", func() framework.Profile { return streaming.NewProfile() }, framework.ProfileCapabilities{})
}

func register(
	name string,
	factory func() framework.Profile,
	capabilities framework.ProfileCapabilities,
) {
	framework.MustRegisterProfile(framework.ProfileRegistration{
		Name:         name,
		Factory:      factory,
		Capabilities: capabilities,
	})
}
