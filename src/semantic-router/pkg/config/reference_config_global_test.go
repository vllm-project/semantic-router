package config

import (
	"reflect"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func assertReferenceConfigRouterGlobalCoverage(t testingT, router map[string]interface{}) {
	modelSelection := mustMapAt(t, router, "model_selection")
	learning := mustMapAt(t, router, "learning")

	assertMapCoversStructFields(t, router, reflect.TypeOf(CanonicalRouterGlobal{}), "global.router")
	assertMapCoversStructFields(t, mustMapAt(t, router, "fallback"), reflect.TypeOf(fallback.FallbackPolicy{}), "global.router.fallback")
	assertMapCoversStructFields(t, mustMapAt(t, router, "fallback", "circuit_breaker"), reflect.TypeOf(fallback.CircuitBreakerConfig{}), "global.router.fallback.circuit_breaker")
	assertMapCoversStructFields(t, mustMapAt(t, router, "streamed_body"), reflect.TypeOf(CanonicalStreamedBody{}), "global.router.streamed_body")
	assertMapCoversStructFields(t, mustMapAt(t, router, "skip_processing"), reflect.TypeOf(SkipProcessingConfig{}), "global.router.skip_processing")
	assertMapCoversStructFields(t, mustMapAt(t, router, "headers"), reflect.TypeOf(RouterHeadersConfig{}), "global.router.headers")
	assertMapCoversStructFields(t, mustMapAt(t, router, "headers", "debug_principals"), reflect.TypeOf(DebugPrincipals{}), "global.router.headers.debug_principals")
	assertMapCoversStructFields(t, mustMapAt(t, router, "routing_input"), reflect.TypeOf(RoutingInputConfig{}), "global.router.routing_input")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, router, "routing_input", "strip_blocks"), reflect.TypeOf(RoutingInputStripBlock{}), "global.router.routing_input.strip_blocks")
	assertMapCoversStructFields(t, modelSelection, reflect.TypeOf(ModelSelectionConfig{}), "global.router.model_selection")
	assertReferenceConfigRouterLearningCoverage(t, learning)
}

func assertReferenceConfigRouterLearningCoverage(t testingT, learning map[string]interface{}) {
	adaptation := mustMapAt(t, learning, "adaptation")
	protection := mustMapAt(t, learning, "protection")
	identity := mustMapAt(t, protection, "identity")
	identityHeaders := mustMapAt(t, identity, "headers")

	assertMapCoversStructFields(t, learning, reflect.TypeOf(RouterLearningConfig{}), "global.router.learning")
	assertMapCoversStructFields(t, adaptation, reflect.TypeOf(RouterLearningAdaptationConfig{}), "global.router.learning.adaptation")
	assertMapCoversStructFields(t, protection, reflect.TypeOf(RouterLearningProtectionConfig{}), "global.router.learning.protection")
	assertMapCoversStructFields(t, identity, reflect.TypeOf(RouterLearningIdentityConfig{}), "global.router.learning.protection.identity")
	assertMapCoversStructFields(t, identityHeaders, reflect.TypeOf(RouterLearningIdentityHeadersConfig{}), "global.router.learning.protection.identity.headers")
	assertMapCoversStructFields(t, mustMapAt(t, protection, "tuning"), reflect.TypeOf(RouterLearningProtectionTuning{}), "global.router.learning.protection.tuning")
	assertMapCoversStructFields(t, mustMapAt(t, protection, "release"), reflect.TypeOf(RouterLearningProtectionReleaseConfig{}), "global.router.learning.protection.release")
}

func assertReferenceConfigServiceGlobalCoverage(t testingT, services map[string]interface{}) {
	assertMapCoversStructFields(t, services, reflect.TypeOf(CanonicalServiceGlobal{}), "global.services")
	assertReferenceConfigAPIServiceCoverage(t, mustMapAt(t, services, "api"))
	assertReferenceConfigResponseAPIServiceCoverage(t, mustMapAt(t, services, "response_api"))
	assertReferenceConfigObservabilityCoverage(t, mustMapAt(t, services, "observability"))
	assertReferenceConfigAuthzCoverage(t, mustMapAt(t, services, "authz"))
	assertReferenceConfigRateLimitCoverage(t, mustMapAt(t, services, "ratelimit"))
	assertReferenceConfigManagementAPICoverage(t, mustMapAt(t, services, "management_api"))
	assertReferenceConfigRouterReplayCoverage(t, mustMapAt(t, services, "router_replay"))
}

func assertReferenceConfigManagementAPICoverage(t testingT, managementAPI map[string]interface{}) {
	assertMapCoversStructFields(t, managementAPI, reflect.TypeOf(ManagementAPIConfig{}), "global.services.management_api")
	assertMapCoversStructFields(t, mustMapAt(t, managementAPI, "auth"), reflect.TypeOf(ManagementAPIAuthConfig{}), "global.services.management_api.auth")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, managementAPI, "auth", "tokens"),
		reflect.TypeOf(ManagementAPITokenRef{}),
		"global.services.management_api.auth.tokens",
	)
	assertReferenceConfigManagementAPIRolesAlign(t, mustMapAt(t, managementAPI, "auth", "roles"))
}

// assertReferenceConfigManagementAPIRolesAlign keeps the exhaustive reference
// YAML role lists in sync with DefaultManagementAPIRoles so copying the sample
// into a bearer deployment does not silently weaken operator permissions.
func assertReferenceConfigManagementAPIRolesAlign(t testingT, roles map[string]interface{}) {
	defaults := DefaultManagementAPIRoles()
	for role, wantPerms := range defaults {
		raw, ok := roles[role]
		if !ok {
			t.Fatalf("global.services.management_api.auth.roles missing %q", role)
		}
		gotList, ok := raw.([]interface{})
		if !ok {
			t.Fatalf("global.services.management_api.auth.roles.%s must be a list", role)
		}
		got := make(map[string]struct{}, len(gotList))
		for _, item := range gotList {
			perm, ok := item.(string)
			if !ok {
				t.Fatalf("global.services.management_api.auth.roles.%s entries must be strings", role)
			}
			got[perm] = struct{}{}
		}
		for _, want := range wantPerms {
			if _, ok := got[want]; !ok {
				t.Fatalf("global.services.management_api.auth.roles.%s missing permission %q (must match DefaultManagementAPIRoles)", role, want)
			}
		}
	}
}

func assertReferenceConfigAPIServiceCoverage(t testingT, api map[string]interface{}) {
	metrics := mustMapAt(t, api, "batch_classification", "metrics")

	assertMapCoversStructFields(t, api, reflect.TypeOf(APIConfig{}), "global.services.api")
	assertMapCoversStructFields(t, mustMapAt(t, api, "routing_preview"), reflect.TypeOf(RoutingPreviewConfig{}), "global.services.api.routing_preview")
	assertMapCoversStructFields(t, mustMapAt(t, api, "batch_classification"), reflect.TypeOf(BatchClassificationConfig{}), "global.services.api.batch_classification", "concurrency_threshold", "max_concurrency")
	assertMapCoversStructFields(t, metrics, reflect.TypeOf(BatchClassificationMetricsConfig{}), "global.services.api.batch_classification.metrics")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, metrics, "batch_size_ranges"),
		reflect.TypeOf(BatchSizeRangeConfig{}),
		"global.services.api.batch_classification.metrics.batch_size_ranges",
	)
}

func assertReferenceConfigResponseAPIServiceCoverage(t testingT, responseAPI map[string]interface{}) {
	assertMapCoversStructFields(t, responseAPI, reflect.TypeOf(ResponseAPIConfig{}), "global.services.response_api")
	assertMapCoversStructFields(t, mustMapAt(t, responseAPI, "redis"), reflect.TypeOf(ResponseAPIRedisConfig{}), "global.services.response_api.redis")
}

func assertReferenceConfigObservabilityCoverage(t testingT, observability map[string]interface{}) {
	tracing := mustMapAt(t, observability, "tracing")

	assertMapCoversStructFields(t, observability, reflect.TypeOf(ObservabilityConfig{}), "global.services.observability")
	assertMapCoversStructFields(t, tracing, reflect.TypeOf(TracingConfig{}), "global.services.observability.tracing")
	assertMapCoversStructFields(t, mustMapAt(t, tracing, "exporter"), reflect.TypeOf(TracingExporterConfig{}), "global.services.observability.tracing.exporter")
	assertMapCoversStructFields(t, mustMapAt(t, tracing, "sampling"), reflect.TypeOf(TracingSamplingConfig{}), "global.services.observability.tracing.sampling")
	assertMapCoversStructFields(t, mustMapAt(t, tracing, "resource"), reflect.TypeOf(TracingResourceConfig{}), "global.services.observability.tracing.resource")
	assertMapCoversStructFields(t, mustMapAt(t, observability, "metrics"), reflect.TypeOf(MetricsConfig{}), "global.services.observability.metrics")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, observability, "metrics", "windowed_metrics"),
		reflect.TypeOf(WindowedMetricsConfig{}),
		"global.services.observability.metrics.windowed_metrics",
	)
	assertMapCoversStructFields(
		t,
		mustMapAt(t, observability, "profiling"),
		reflect.TypeOf(ProfilingConfig{}),
		"global.services.observability.profiling",
	)
}

func assertReferenceConfigAuthzCoverage(t testingT, authz map[string]interface{}) {
	assertMapCoversStructFields(t, authz, reflect.TypeOf(AuthzConfig{}), "global.services.authz")
	assertMapCoversStructFields(t, mustMapAt(t, authz, "identity"), reflect.TypeOf(IdentityConfig{}), "global.services.authz.identity")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, authz, "providers"),
		reflect.TypeOf(AuthzProviderConfig{}),
		"global.services.authz.providers",
	)
}

func assertReferenceConfigRateLimitCoverage(t testingT, ratelimit map[string]interface{}) {
	providers := mustSliceAt(t, ratelimit, "providers")
	rules := collectNestedSliceItems(t, providers, "rules", "global.services.ratelimit.providers")

	assertMapCoversStructFields(t, ratelimit, reflect.TypeOf(RateLimitConfig{}), "global.services.ratelimit")
	assertSliceUnionCoversStructFields(t, providers, reflect.TypeOf(RateLimitProviderConfig{}), "global.services.ratelimit.providers")
	assertSliceUnionCoversStructFields(t, rules, reflect.TypeOf(RateLimitRule{}), "global.services.ratelimit.providers[].rules")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, rules, "match", "global.services.ratelimit.providers[].rules"),
		reflect.TypeOf(RateLimitMatch{}),
		"global.services.ratelimit.providers[].rules[].match",
	)
}

func assertReferenceConfigRouterReplayCoverage(t testingT, routerReplay map[string]interface{}) {
	assertMapCoversStructFields(t, routerReplay, reflect.TypeOf(RouterReplayConfig{}), "global.services.router_replay")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "redis"), reflect.TypeOf(RouterReplayRedisConfig{}), "global.services.router_replay.redis")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "postgres"), reflect.TypeOf(RouterReplayPostgresConfig{}), "global.services.router_replay.postgres")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "milvus"), reflect.TypeOf(RouterReplayMilvusConfig{}), "global.services.router_replay.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "qdrant"), reflect.TypeOf(RouterReplayQdrantConfig{}), "global.services.router_replay.qdrant")
}

func assertReferenceConfigStoreGlobalCoverage(t testingT, stores map[string]interface{}) {
	assertMapCoversStructFields(t, stores, reflect.TypeOf(CanonicalStoreGlobal{}), "global.stores")
	assertReferenceConfigSemanticCacheCoverage(t, mustMapAt(t, stores, "response_cache"))
	assertReferenceConfigMemoryCoverage(t, mustMapAt(t, stores, "memory"))
	assertReferenceConfigVectorStoreCoverage(t, mustMapAt(t, stores, "vector_store"))
	assertReferenceConfigToolSessionsCoverage(t, mustMapAt(t, stores, "tool_sessions"))
}

func assertReferenceConfigToolSessionsCoverage(t testingT, toolSessions map[string]interface{}) {
	assertMapCoversStructFields(t, toolSessions, reflect.TypeOf(ToolSessionStoreConfig{}), "global.stores.tool_sessions")
	assertMapCoversStructFields(t, mustMapAt(t, toolSessions, "redis"), reflect.TypeOf(ToolSessionRedisConfig{}), "global.stores.tool_sessions.redis")
}

func assertReferenceConfigSemanticCacheCoverage(t testingT, semanticCache map[string]interface{}) {
	assertMapCoversStructFields(t, semanticCache, reflect.TypeOf(responseCacheStoreReference{}), "global.stores.response_cache")
	assertMapCoversStructFields(t, mustMapAt(t, semanticCache, "milvus"), reflect.TypeOf(MilvusConfig{}), "global.stores.response_cache.milvus")
}

type responseCacheStoreReference struct {
	BackendType         string            `yaml:"backend_type,omitempty"`
	Enabled             bool              `yaml:"enabled"`
	SimilarityThreshold *float32          `yaml:"similarity_threshold,omitempty"`
	MaxEntries          int               `yaml:"max_entries,omitempty"`
	TTLSeconds          int               `yaml:"ttl_seconds,omitempty"`
	EvictionPolicy      string            `yaml:"eviction_policy,omitempty"`
	Milvus              *MilvusConfig     `yaml:"milvus,omitempty"`
	Embedding           *EmbeddingBinding `yaml:"embedding,omitempty"`
}

func assertReferenceConfigMemoryCoverage(t testingT, memory map[string]interface{}) {
	assertMapCoversStructFields(t, memory, reflect.TypeOf(MemoryConfig{}), "global.stores.memory")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "milvus"), reflect.TypeOf(MemoryMilvusConfig{}), "global.stores.memory.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "valkey"), reflect.TypeOf(MemoryValkeyConfig{}), "global.stores.memory.valkey")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "qdrant"), reflect.TypeOf(MemoryQdrantConfig{}), "global.stores.memory.qdrant")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "reflection"), reflect.TypeOf(MemoryReflectionConfig{}), "global.stores.memory.reflection")
}

func assertReferenceConfigVectorStoreCoverage(t testingT, vectorStore map[string]interface{}) {
	assertMapCoversStructFields(t, vectorStore, reflect.TypeOf(VectorStoreConfig{}), "global.stores.vector_store")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "memory"), reflect.TypeOf(VectorStoreMemoryConfig{}), "global.stores.vector_store.memory")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "llama_stack"), reflect.TypeOf(LlamaStackVectorStoreConfig{}), "global.stores.vector_store.llama_stack")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "milvus"), reflect.TypeOf(MilvusConfig{}), "global.stores.vector_store.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "valkey"), reflect.TypeOf(ValkeyVectorStoreConfig{}), "global.stores.vector_store.valkey")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "qdrant"), reflect.TypeOf(QdrantVectorStoreConfig{}), "global.stores.vector_store.qdrant")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "metadata_postgres"), reflect.TypeOf(VectorStoreMetadataPostgresConfig{}), "global.stores.vector_store.metadata_postgres")
}

func assertReferenceConfigIntegrationGlobalCoverage(t testingT, integrations map[string]interface{}) {
	assertMapCoversStructFields(t, integrations, reflect.TypeOf(CanonicalIntegrationGlobal{}), "global.integrations")
	assertMapCoversStructFields(t, mustMapAt(t, integrations, "looper"), reflect.TypeOf(LooperConfig{}), "global.integrations.looper", "endpoint")
	flow := mustMapAt(t, integrations, "looper", "flow")
	assertMapCoversStructFields(t, flow, reflect.TypeOf(FlowRuntimeConfig{}), "global.integrations.looper.flow")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state"), reflect.TypeOf(WorkflowStateRuntimeConfig{}), "global.integrations.looper.flow.state")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state", "file"), reflect.TypeOf(WorkflowStateFileConfig{}), "global.integrations.looper.flow.state.file")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state", "redis"), reflect.TypeOf(WorkflowStateRedisConfig{}), "global.integrations.looper.flow.state.redis")
}

func assertReferenceConfigModelCatalogCoverage(t testingT, modelCatalog map[string]interface{}) {
	// external[] is the deprecated way to declare a remote model: a
	// deployment with a connection block replaces it.
	assertMapCoversStructFields(t, modelCatalog, reflect.TypeOf(CanonicalModelCatalog{}), "global.model_catalog", "external")
	assertReferenceConfigDeploymentCoverage(t, mustMapAt(t, modelCatalog, "deployments"))
	assertReferenceConfigBindingsBySignalType(t, mustMapAt(t, modelCatalog, "bindings"), "global.model_catalog.bindings")
	assertReferenceConfigEmbeddingCatalogCoverage(t, mustMapAt(t, modelCatalog, "embeddings"))
	// decision_model is the only system line; the others are the deprecated
	// way to choose a signal's model, which a binding now names.
	assertMapCoversStructFields(t, mustMapAt(t, modelCatalog, "system"), reflect.TypeOf(CanonicalSystemModels{}), "global.model_catalog.system",
		"prompt_guard", "pii_classifier", "hallucination_detector")
	assertReferenceConfigModelModuleCoverage(t, mustMapAt(t, modelCatalog, "modules"))
}

// assertReferenceConfigDeploymentCoverage requires, across the reference
// deployments, every deployment field (the managed model_runtime fields and
// the connection block of an attached runtime and the remote providers),
// every field of their nested blocks, and every provider, with
// model_runtime both managed and attached.
func assertReferenceConfigDeploymentCoverage(t testingT, deployments map[string]interface{}) {
	items := referenceMapValues(deployments)
	// external_model, question_types and served_name are the deprecated
	// spellings of a remote provider, capabilities.question_types and model.
	assertSliceUnionCoversStructFields(t, items, reflect.TypeOf(ModelDeployment{}), "global.model_catalog.deployments", "external_model", "question_types", "served_name")
	for _, block := range []struct {
		key string
		typ reflect.Type
	}{
		{"input", reflect.TypeOf(ModelInputBudget{})},
		{"auth", reflect.TypeOf(DeploymentAuth{})},
		{"tls", reflect.TypeOf(DeploymentTLS{})},
		{"limits", reflect.TypeOf(DeploymentLimits{})},
		{"capabilities", reflect.TypeOf(DeploymentCapabilities{})},
		{"generation", reflect.TypeOf(DeploymentGeneration{})},
		{"operating_point", reflect.TypeOf(OperatingPointReference{})},
	} {
		assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, items, block.key, "global.model_catalog.deployments"), block.typ, "global.model_catalog.deployments[]."+block.key)
	}
	assertSliceUnionCoversStructFields(t, collectNestedSliceItems(t, items, "replicas", "global.model_catalog.deployments"), reflect.TypeOf(ModelReplica{}), "global.model_catalog.deployments[].replicas", "served_name")

	providers := map[string]bool{}
	for _, item := range items {
		deployment := mustMapValue(t, item, "global.model_catalog.deployments")
		provider := mustStringAt(t, deployment, "provider")
		if provider == ModelRuntimeProvider {
			if _, attached := deployment["endpoint"]; attached {
				provider += " (attached)"
			}
		}
		providers[provider] = true
	}
	for _, provider := range []string{ModelRuntimeProvider, ModelRuntimeProvider + " (attached)", SystemOneProvider, ClassifyProvider, OpenAIChatProvider, OpenAIEmbeddingsProvider} {
		if !providers[provider] {
			t.Fatalf("global.model_catalog.deployments must declare a %s deployment", provider)
		}
	}
}

// assertReferenceConfigBindingsBySignalType checks that a bindings map names
// deployments, by name, for signal types whose rules share one answer.
func assertReferenceConfigBindingsBySignalType(t testingT, bindings map[string]interface{}, path string) {
	if len(bindings) == 0 {
		t.Fatalf("%s must bind at least one signal type", path)
	}
	for key, value := range bindings {
		if !slices.Contains(SharedAnswerSignalTypes, key) {
			t.Fatalf("%s.%s: a binding key is a signal type whose rules share one answer (%v)", path, key, SharedAnswerSignalTypes)
		}
		if name, ok := value.(string); !ok || name == "" {
			t.Fatalf("%s.%s must name a deployment, got %#v", path, key, value)
		}
	}
}

func referenceMapValues(node map[string]interface{}) []interface{} {
	keys := make([]string, 0, len(node))
	for key := range node {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	values := make([]interface{}, 0, len(keys))
	for _, key := range keys {
		values = append(values, node[key])
	}
	return values
}

func assertReferenceConfigEmbeddingCatalogCoverage(t testingT, embeddings map[string]interface{}) {
	assertMapCoversStructFields(t, embeddings, reflect.TypeOf(CanonicalEmbeddingModels{}), "global.model_catalog.embeddings")
}

func assertReferenceConfigModelModuleCoverage(t testingT, modules map[string]interface{}) {
	assertMapCoversStructFields(t, modules, reflect.TypeOf(CanonicalModelModules{}), "global.model_catalog.modules")
	assertMapCoversStructFields(t, mustMapAt(t, modules, "prompt_compression"), reflect.TypeOf(PromptCompressionConfig{}), "global.model_catalog.modules.prompt_compression")
}
