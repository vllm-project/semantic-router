package all

import (
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	authzrbac "github.com/vllm-project/semantic-router/e2e/profiles/authz-rbac"
	dashboard "github.com/vllm-project/semantic-router/e2e/profiles/dashboard"
	delegatedcredential "github.com/vllm-project/semantic-router/e2e/profiles/delegated-credential"
	dynamicconfig "github.com/vllm-project/semantic-router/e2e/profiles/dynamic-config"
	externalgatewayresponses "github.com/vllm-project/semantic-router/e2e/profiles/external-gateway-responses"
	hallucination "github.com/vllm-project/semantic-router/e2e/profiles/hallucination"
	imagerouting "github.com/vllm-project/semantic-router/e2e/profiles/image-routing"
	jailbreakonerror "github.com/vllm-project/semantic-router/e2e/profiles/jailbreak-onerror"
	looper "github.com/vllm-project/semantic-router/e2e/profiles/looper"
	masking "github.com/vllm-project/semantic-router/e2e/profiles/masking"
	modelruntime "github.com/vllm-project/semantic-router/e2e/profiles/model-runtime"
	modelruntimereal "github.com/vllm-project/semantic-router/e2e/profiles/model-runtime-real"
	modelruntimevela3 "github.com/vllm-project/semantic-router/e2e/profiles/model-runtime-vela3"
	multiendpoint "github.com/vllm-project/semantic-router/e2e/profiles/multi-endpoint"
	piiremotebackend "github.com/vllm-project/semantic-router/e2e/profiles/pii-remote-backend"
	productionstack "github.com/vllm-project/semantic-router/e2e/profiles/production-stack"
	progressgate "github.com/vllm-project/semantic-router/e2e/profiles/progress-gate"
	providerprotocols "github.com/vllm-project/semantic-router/e2e/profiles/provider-protocols"
	raghybridsearch "github.com/vllm-project/semantic-router/e2e/profiles/rag-hybrid-search"
	remoteembedding "github.com/vllm-project/semantic-router/e2e/profiles/remote-embedding"
	responseapi "github.com/vllm-project/semantic-router/e2e/profiles/response-api"
	responseapiredis "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis"
	responseapirediscluster "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis-cluster"
	routeaction "github.com/vllm-project/semantic-router/e2e/profiles/route-action"
	routerreplay "github.com/vllm-project/semantic-router/e2e/profiles/router-replay"
	routingstrategies "github.com/vllm-project/semantic-router/e2e/profiles/routing-strategies"
	signalrouting "github.com/vllm-project/semantic-router/e2e/profiles/signal-routing"
	standalone "github.com/vllm-project/semantic-router/e2e/profiles/standalone"
	streaming "github.com/vllm-project/semantic-router/e2e/profiles/streaming"
	vectorstoreregistry "github.com/vllm-project/semantic-router/e2e/profiles/vectorstore-registry"
)

var providerMockerLocalImages = []framework.LocalImageBuild{
	{
		Dockerfile:   "tools/test/services/provider-mocker/Dockerfile",
		Tag:          "semantic-router-ci/provider-mocker:e2e-test",
		BuildContext: "tools/test/services/provider-mocker",
		RolloutRestarts: []framework.RolloutRestartTarget{
			{Namespace: "default", Deployment: "provider-mocker"},
			{Namespace: "provider-protocols-system", Deployment: "provider-mocker"},
			{Namespace: "delegated-credential-system", Deployment: "delegated-gateway"},
			{Namespace: "delegated-credential-system", Deployment: "static-backend"},
			{Namespace: "default", Deployment: "looper-provider-mocker"},
			{Namespace: "default", Deployment: "mock-llm"},
		},
	},
}

// modelRuntimeLocalImages adds the CPU model runtime image to the provider
// fixtures for profiles that run a runtime outside the Router; it writes its
// tiny fixture models at start-up.
var modelRuntimeLocalImages = append(append([]framework.LocalImageBuild(nil), providerMockerLocalImages...),
	framework.LocalImageBuild{
		Dockerfile:   "src/model-runtime/Dockerfile",
		Tag:          "semantic-router-ci/model-runtime:e2e-test",
		BuildContext: "src/model-runtime",
		RolloutRestarts: []framework.RolloutRestartTarget{
			{Namespace: "vllm-semantic-router-system", Deployment: "model-runtime-attached"},
		},
	},
)

var dashboardLocalImages = []framework.LocalImageBuild{
	{
		Dockerfile:   "dashboard/backend/Dockerfile",
		Tag:          "ghcr.io/vllm-project/semantic-router/dashboard:e2e-test",
		BuildContext: ".",
	},
}

func init() {
	register(
		"envoy-ai-gateway",
		func() framework.Profile { return aigateway.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"provider-protocols",
		func() framework.Profile { return providerprotocols.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("authz-rbac", func() framework.Profile { return authzrbac.NewProfile() }, framework.ProfileCapabilities{})
	register(
		"delegated-credential",
		func() framework.Profile { return delegatedcredential.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("pii-remote-backend", func() framework.Profile { return piiremotebackend.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("model-runtime", func() framework.Profile { return modelruntime.NewProfile() }, framework.ProfileCapabilities{LocalImages: modelRuntimeLocalImages})
	register("model-runtime-real", func() framework.Profile { return modelruntimereal.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("model-runtime-vela3", func() framework.Profile { return modelruntimevela3.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register(
		"dashboard",
		func() framework.Profile { return dashboard.NewProfile() },
		framework.ProfileCapabilities{LocalImages: dashboardLocalImages},
	)
	register("dynamic-config", func() framework.Profile { return dynamicconfig.NewProfile() }, framework.ProfileCapabilities{})
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
	register(
		"image-routing",
		func() framework.Profile { return imagerouting.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register(
		"jailbreak-onerror",
		func() framework.Profile { return jailbreakonerror.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("masking", func() framework.Profile { return masking.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register(
		"route-action",
		func() framework.Profile { return routeaction.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("looper", func() framework.Profile { return looper.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("multi-endpoint", func() framework.Profile { return multiendpoint.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("production-stack", func() framework.Profile { return productionstack.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
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
		"progress-gate",
		func() framework.Profile { return progressgate.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("remote-embedding", func() framework.Profile { return remoteembedding.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register(
		"router-replay",
		func() framework.Profile { return routerreplay.NewProfile() },
		framework.ProfileCapabilities{LocalImages: providerMockerLocalImages},
	)
	register("routing-strategies", func() framework.Profile { return routingstrategies.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("signal-routing", func() framework.Profile { return signalrouting.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("standalone", func() framework.Profile { return standalone.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
	register("streaming", func() framework.Profile { return streaming.NewProfile() }, framework.ProfileCapabilities{LocalImages: providerMockerLocalImages})
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
