package all

import (
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	agentgateway "github.com/vllm-project/semantic-router/e2e/profiles/agentgateway"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	aibrix "github.com/vllm-project/semantic-router/e2e/profiles/aibrix"
	authzrbac "github.com/vllm-project/semantic-router/e2e/profiles/authz-rbac"
	categoryremotebackend "github.com/vllm-project/semantic-router/e2e/profiles/category-remote-backend"
	complexityremotebackend "github.com/vllm-project/semantic-router/e2e/profiles/complexity-remote-backend"
	dashboard "github.com/vllm-project/semantic-router/e2e/profiles/dashboard"
	dynamicconfig "github.com/vllm-project/semantic-router/e2e/profiles/dynamic-config"
	dynamo "github.com/vllm-project/semantic-router/e2e/profiles/dynamo"
	externalgatewayresponses "github.com/vllm-project/semantic-router/e2e/profiles/external-gateway-responses"
	hallucination "github.com/vllm-project/semantic-router/e2e/profiles/hallucination"
	istio "github.com/vllm-project/semantic-router/e2e/profiles/istio"
	jailbreakonerror "github.com/vllm-project/semantic-router/e2e/profiles/jailbreak-onerror"
	llmd "github.com/vllm-project/semantic-router/e2e/profiles/llm-d"
	localclassifierbackend "github.com/vllm-project/semantic-router/e2e/profiles/local-classifier-backend"
	looper "github.com/vllm-project/semantic-router/e2e/profiles/looper"
	mlmodelselection "github.com/vllm-project/semantic-router/e2e/profiles/ml-model-selection"
	multiendpoint "github.com/vllm-project/semantic-router/e2e/profiles/multi-endpoint"
	multimodalrouting "github.com/vllm-project/semantic-router/e2e/profiles/multimodal-routing"
	piiremotebackend "github.com/vllm-project/semantic-router/e2e/profiles/pii-remote-backend"
	productionstack "github.com/vllm-project/semantic-router/e2e/profiles/production-stack"
	progressgate "github.com/vllm-project/semantic-router/e2e/profiles/progress-gate"
	providerprotocols "github.com/vllm-project/semantic-router/e2e/profiles/provider-protocols"
	raghybridsearch "github.com/vllm-project/semantic-router/e2e/profiles/rag-hybrid-search"
	remoteembedding "github.com/vllm-project/semantic-router/e2e/profiles/remote-embedding"
	responseapi "github.com/vllm-project/semantic-router/e2e/profiles/response-api"
	responseapiredis "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis"
	responseapirediscluster "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis-cluster"
	responsejailbreak "github.com/vllm-project/semantic-router/e2e/profiles/response-jailbreak"
	routeaction "github.com/vllm-project/semantic-router/e2e/profiles/route-action"
	routerreplay "github.com/vllm-project/semantic-router/e2e/profiles/router-replay"
	routingstrategies "github.com/vllm-project/semantic-router/e2e/profiles/routing-strategies"
	stickytoolselectionexpiry "github.com/vllm-project/semantic-router/e2e/profiles/sticky-tool-selection-expiry"
	stickytoolselectionredis "github.com/vllm-project/semantic-router/e2e/profiles/sticky-tool-selection-redis"
	streaming "github.com/vllm-project/semantic-router/e2e/profiles/streaming"
	vectorstoreregistry "github.com/vllm-project/semantic-router/e2e/profiles/vectorstore-registry"
	velahalu "github.com/vllm-project/semantic-router/e2e/profiles/vela-halu"
	velaomni "github.com/vllm-project/semantic-router/e2e/profiles/vela-omni"
)

var providerMockerLocalImages = []framework.LocalImageBuild{
	{
		Dockerfile:   "tools/test/services/provider-mocker/Dockerfile",
		Tag:          "semantic-router-ci/provider-mocker:e2e-test",
		BuildContext: "tools/test/services/provider-mocker",
		RolloutRestarts: []framework.RolloutRestartTarget{
			{Namespace: "default", Deployment: "provider-mocker"},
			{Namespace: "provider-protocols-system", Deployment: "provider-mocker"},
			{Namespace: "default", Deployment: "looper-provider-mocker"},
			{Namespace: "default", Deployment: "mock-llm"},
		},
	},
}

var dashboardLocalImages = []framework.LocalImageBuild{
	{
		Dockerfile:   "dashboard/backend/Dockerfile",
		Tag:          "ghcr.io/vllm-project/semantic-router/dashboard:e2e-test",
		BuildContext: ".",
	},
}

func init() {
	register("vela-halu", func() framework.Profile { return velahalu.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("agentgateway", func() framework.Profile { return agentgateway.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"envoy-ai-gateway",
		func() framework.Profile { return aigateway.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("aibrix", func() framework.Profile { return aibrix.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"provider-protocols",
		func() framework.Profile { return providerprotocols.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("authz-rbac", func() framework.Profile { return authzrbac.NewProfile() }, framework.ProfileCapabilities{})
	register("category-remote-backend", func() framework.Profile { return categoryremotebackend.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("complexity-remote-backend", func() framework.Profile { return complexityremotebackend.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("pii-remote-backend", func() framework.Profile { return piiremotebackend.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("local-classifier-backend", func() framework.Profile { return localclassifierbackend.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register(
		"dashboard",
		func() framework.Profile { return dashboard.NewProfile() },
		framework.ProfileCapabilities{LocalImages: dashboardLocalImages},
	)
	register("dynamic-config", func() framework.Profile { return dynamicconfig.NewProfile() }, framework.ProfileCapabilities{})
	register("dynamo", func() framework.Profile { return dynamo.NewProfile() }, framework.ProfileCapabilities{RequiresGPU: true})
	register(
		"external-gateway-responses",
		func() framework.Profile { return externalgatewayresponses.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"hallucination",
		func() framework.Profile { return hallucination.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("istio", func() framework.Profile { return istio.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"jailbreak-onerror",
		func() framework.Profile { return jailbreakonerror.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("llm-d", func() framework.Profile { return llmd.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"route-action",
		func() framework.Profile { return routeaction.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("looper", func() framework.Profile { return looper.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register(
		"ml-model-selection",
		func() framework.Profile { return mlmodelselection.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("multi-endpoint", func() framework.Profile { return multiendpoint.NewProfile() }, framework.ProfileCapabilities{})
	register("vela-omni", func() framework.Profile { return velaomni.NewProfile() }, framework.ProfileCapabilities{
		LocalImages:     providerMockerLocalImages,
		RouterBuildArgs: map[string]string{"VELA_OMNI_VARIANTS": "nano mini"},
	})
	register("multimodal-routing", func() framework.Profile { return multimodalrouting.NewProfile() }, framework.ProfileCapabilities{})
	register("production-stack", func() framework.Profile { return productionstack.NewProfile() }, framework.ProfileCapabilities{})
	register("rag-hybrid-search", func() framework.Profile { return raghybridsearch.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"response-api",
		func() framework.Profile { return responseapi.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"response-api-redis",
		func() framework.Profile { return responseapiredis.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"response-api-redis-cluster",
		func() framework.Profile { return responseapirediscluster.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"response-jailbreak",
		func() framework.Profile { return responsejailbreak.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"progress-gate",
		func() framework.Profile { return progressgate.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("remote-embedding", func() framework.Profile { return remoteembedding.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"router-replay",
		func() framework.Profile { return routerreplay.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("routing-strategies", func() framework.Profile { return routingstrategies.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"sticky-tool-selection-expiry",
		func() framework.Profile { return stickytoolselectionexpiry.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register(
		"sticky-tool-selection-redis",
		func() framework.Profile { return stickytoolselectionredis.NewProfile() },
		framework.ProfileCapabilities{LocalImages: mockVLLMLocalImages},
	)
	register("streaming", func() framework.Profile { return streaming.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"vectorstore-registry",
		func() framework.Profile { return vectorstoreregistry.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
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
