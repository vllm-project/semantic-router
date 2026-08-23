package config

import "reflect"

func assertReferenceConfigRouterGlobalCoverage(t testingT, router map[string]interface{}) {
	learning := mustMapAt(t, router, "learning")

	assertMapCoversStructFields(t, router, reflect.TypeOf(CanonicalRouterGlobal{}), "global.router")
	assertMapCoversStructFields(t, mustMapAt(t, router, "streamed_body"), reflect.TypeOf(CanonicalStreamedBody{}), "global.router.streamed_body")
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
}

func assertReferenceConfigServiceGlobalCoverage(t testingT, services map[string]interface{}) {
	assertMapCoversStructFields(
		t,
		services,
		reflect.TypeOf(CanonicalServiceGlobal{}),
		"global.services",
		"access",
		"backend_credentials",
		"backend_egress",
	)
	assertReferenceConfigAPIServiceCoverage(t, mustMapAt(t, services, "api"))
	assertReferenceConfigResponseAPIServiceCoverage(t, mustMapAt(t, services, "response_api"))
	assertMapCoversStructFields(t, mustMapAt(t, services, "agent"), reflect.TypeOf(AgentServiceConfig{}), "global.services.agent")
	assertReferenceConfigObservabilityCoverage(t, mustMapAt(t, services, "observability"))
	assertReferenceConfigManagementAPICoverage(t, mustMapAt(t, services, "management_api"))
	assertReferenceConfigRouterReplayCoverage(t, mustMapAt(t, services, "router_replay"))
}

func assertReferenceConfigManagementAPICoverage(t testingT, managementAPI map[string]interface{}) {
	assertMapCoversStructFields(t, managementAPI, reflect.TypeOf(ManagementAPIConfig{}), "global.services.management_api", "tls")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, managementAPI, "auth"),
		reflect.TypeOf(ManagementAPIAuthConfig{}),
		"global.services.management_api.auth",
		"token_signing_keyring_file",
		"token_signing_keyring_env",
		"service_account_hmac_keyring_file",
		"service_account_hmac_keyring_env",
		"invitation_hmac_keyring_file",
		"invitation_hmac_keyring_env",
		"response_kek_keyring_file",
		"response_kek_keyring_env",
		"bootstrap",
		"recovery",
	)
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
	assertMapCoversStructFields(t, mustMapAt(t, api, "batch_classification"), reflect.TypeOf(BatchClassificationConfig{}), "global.services.api.batch_classification")
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
}

func assertReferenceConfigRouterReplayCoverage(t testingT, routerReplay map[string]interface{}) {
	assertMapCoversStructFields(t, routerReplay, reflect.TypeOf(RouterReplayConfig{}), "global.services.router_replay")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "redis"), reflect.TypeOf(RouterReplayRedisConfig{}), "global.services.router_replay.redis")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "postgres"), reflect.TypeOf(RouterReplayPostgresConfig{}), "global.services.router_replay.postgres")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "milvus"), reflect.TypeOf(RouterReplayMilvusConfig{}), "global.services.router_replay.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, routerReplay, "qdrant"), reflect.TypeOf(RouterReplayQdrantConfig{}), "global.services.router_replay.qdrant")
}

func assertReferenceConfigStoreGlobalCoverage(t testingT, stores map[string]interface{}) {
	assertMapCoversStructFields(t, stores, reflect.TypeOf(CanonicalStoreGlobal{}), "global.stores", "access", "access_runtime")
	assertReferenceConfigSemanticCacheCoverage(t, mustMapAt(t, stores, "response_cache"))
	assertReferenceConfigMemoryCoverage(t, mustMapAt(t, stores, "memory"))
	assertReferenceConfigVectorStoreCoverage(t, mustMapAt(t, stores, "vector_store"))
}

func assertReferenceConfigSemanticCacheCoverage(t testingT, semanticCache map[string]interface{}) {
	assertMapCoversStructFields(t, semanticCache, reflect.TypeOf(responseCacheStoreReference{}), "global.stores.response_cache")
	assertMapCoversStructFields(t, mustMapAt(t, semanticCache, "milvus"), reflect.TypeOf(MilvusConfig{}), "global.stores.response_cache.milvus")
}

type responseCacheStoreReference struct {
	BackendType         string        `yaml:"backend_type,omitempty"`
	Enabled             bool          `yaml:"enabled"`
	SimilarityThreshold *float32      `yaml:"similarity_threshold,omitempty"`
	MaxEntries          int           `yaml:"max_entries,omitempty"`
	TTLSeconds          int           `yaml:"ttl_seconds,omitempty"`
	EvictionPolicy      string        `yaml:"eviction_policy,omitempty"`
	Milvus              *MilvusConfig `yaml:"milvus,omitempty"`
	EmbeddingModel      string        `yaml:"embedding_model,omitempty"`
}

func assertReferenceConfigMemoryCoverage(t testingT, memory map[string]interface{}) {
	assertMapCoversStructFields(t, memory, reflect.TypeOf(MemoryConfig{}), "global.stores.memory")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "milvus"), reflect.TypeOf(MemoryMilvusConfig{}), "global.stores.memory.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "valkey"), reflect.TypeOf(MemoryValkeyConfig{}), "global.stores.memory.valkey")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "qdrant"), reflect.TypeOf(MemoryQdrantConfig{}), "global.stores.memory.qdrant")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "quality_scoring"), reflect.TypeOf(MemoryQualityScoringConfig{}), "global.stores.memory.quality_scoring")
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
	tools := mustMapAt(t, integrations, "tools")

	assertMapCoversStructFields(t, integrations, reflect.TypeOf(CanonicalIntegrationGlobal{}), "global.integrations")
	assertMapCoversStructFields(t, tools, reflect.TypeOf(ToolsConfig{}), "global.integrations.tools")
	assertMapCoversStructFields(t, mustMapAt(t, tools, "advanced_filtering"), reflect.TypeOf(AdvancedToolFilteringConfig{}), "global.integrations.tools.advanced_filtering")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, tools, "advanced_filtering", "hybrid_history"),
		reflect.TypeOf(HybridHistoryToolRetrievalConfig{}),
		"global.integrations.tools.advanced_filtering.hybrid_history",
	)
	assertMapCoversStructFields(
		t,
		mustMapAt(t, tools, "advanced_filtering", "weights"),
		reflect.TypeOf(ToolFilteringWeights{}),
		"global.integrations.tools.advanced_filtering.weights",
	)
	assertMapCoversStructFields(t, mustMapAt(t, integrations, "looper"), reflect.TypeOf(LooperConfig{}), "global.integrations.looper")
	assertMapCoversStructFields(t, mustMapAt(t, integrations, "looper", "remom"), reflect.TypeOf(ReMoMRuntimeConfig{}), "global.integrations.looper.remom")
	assertMapCoversStructFields(t, mustMapAt(t, integrations, "looper", "fusion"), reflect.TypeOf(FusionRuntimeConfig{}), "global.integrations.looper.fusion")
	flow := mustMapAt(t, integrations, "looper", "flow")
	assertMapCoversStructFields(t, flow, reflect.TypeOf(FlowRuntimeConfig{}), "global.integrations.looper.flow")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state"), reflect.TypeOf(WorkflowStateRuntimeConfig{}), "global.integrations.looper.flow.state")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state", "file"), reflect.TypeOf(WorkflowStateFileConfig{}), "global.integrations.looper.flow.state.file")
	assertMapCoversStructFields(t, mustMapAt(t, flow, "state", "redis"), reflect.TypeOf(WorkflowStateRedisConfig{}), "global.integrations.looper.flow.state.redis")
}

func assertReferenceConfigModelCatalogCoverage(t testingT, modelCatalog map[string]interface{}) {
	assertMapCoversStructFields(t, modelCatalog, reflect.TypeOf(CanonicalModelCatalog{}), "global.model_catalog")
	assertReferenceConfigEmbeddingCatalogCoverage(t, mustMapAt(t, modelCatalog, "embeddings"))
	assertMapCoversStructFields(t, mustMapAt(t, modelCatalog, "system"), reflect.TypeOf(CanonicalSystemModels{}), "global.model_catalog.system")
	assertReferenceConfigExternalCatalogCoverage(t, mustSliceAt(t, modelCatalog, "external"))
	assertReferenceConfigKnowledgeBaseCoverage(t, mustSliceAt(t, modelCatalog, "kbs"))
	assertReferenceConfigModelModuleCoverage(t, mustMapAt(t, modelCatalog, "modules"))
}

func assertReferenceConfigEmbeddingCatalogCoverage(t testingT, embeddings map[string]interface{}) {
	assertMapCoversStructFields(t, embeddings, reflect.TypeOf(CanonicalEmbeddingModels{}), "global.model_catalog.embeddings")
	assertMapCoversStructFields(t, mustMapAt(t, embeddings, "semantic"), reflect.TypeOf(EmbeddingModels{}), "global.model_catalog.embeddings.semantic")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, embeddings, "semantic", "embedding_config"),
		reflect.TypeOf(HNSWConfig{}),
		"global.model_catalog.embeddings.semantic.embedding_config",
	)
	assertMapCoversStructFields(
		t,
		mustMapAt(t, embeddings, "semantic", "embedding_config", "prototype_scoring"),
		reflect.TypeOf(PrototypeScoringConfig{}),
		"global.model_catalog.embeddings.semantic.embedding_config.prototype_scoring",
	)
	assertMapCoversStructFields(
		t,
		mustMapAt(t, embeddings, "semantic", "endpoint"),
		reflect.TypeOf(EmbeddingEndpointConfig{}),
		"global.model_catalog.embeddings.semantic.endpoint",
	)
}

func assertReferenceConfigExternalCatalogCoverage(t testingT, external []interface{}) {
	assertSliceUnionCoversStructFields(t, external, reflect.TypeOf(ExternalModelConfig{}), "global.model_catalog.external")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, external, "llm_endpoint", "global.model_catalog.external"),
		reflect.TypeOf(ClassifierVLLMEndpoint{}),
		"global.model_catalog.external[].llm_endpoint",
	)
}

func assertReferenceConfigKnowledgeBaseCoverage(t testingT, kbs []interface{}) {
	assertSliceUnionCoversStructFields(t, kbs, reflect.TypeOf(KnowledgeBaseConfig{}), "global.model_catalog.kbs")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, kbs, "source", "global.model_catalog.kbs"),
		reflect.TypeOf(KnowledgeBaseSource{}),
		"global.model_catalog.kbs[].source",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, kbs, "prototype_scoring", "global.model_catalog.kbs"),
		reflect.TypeOf(PrototypeScoringConfig{}),
		"global.model_catalog.kbs[].prototype_scoring",
	)
}

func assertReferenceConfigModelModuleCoverage(t testingT, modules map[string]interface{}) {
	assertMapCoversStructFields(t, modules, reflect.TypeOf(CanonicalModelModules{}), "global.model_catalog.modules")
	assertMapCoversStructFields(t, mustMapAt(t, modules, "prompt_compression"), reflect.TypeOf(PromptCompressionConfig{}), "global.model_catalog.modules.prompt_compression")
	// protocol is mutually exclusive with variant (PromptGuardConfig); the
	// reference config demonstrates the local variant path, so protocol has
	// no reference-config key to cover here.
	assertMapCoversStructFields(t, mustMapAt(t, modules, "prompt_guard"), reflect.TypeOf(CanonicalPromptGuardModule{}), "global.model_catalog.modules.prompt_guard", "protocol")
	assertReferenceConfigClassifierModuleCoverage(t, mustMapAt(t, modules, "classifier"))
	assertReferenceConfigComplexityModuleCoverage(t, mustMapAt(t, modules, "complexity"))
	assertReferenceConfigHallucinationModuleCoverage(t, mustMapAt(t, modules, "hallucination_mitigation"))
	assertMapCoversStructFields(t, mustMapAt(t, modules, "feedback_detector"), reflect.TypeOf(CanonicalFeedbackDetectorModule{}), "global.model_catalog.modules.feedback_detector")
	assertMapCoversStructFields(t, mustMapAt(t, modules, "modality_detector"), reflect.TypeOf(ModalityDetectorConfig{}), "global.model_catalog.modules.modality_detector")
}

func assertReferenceConfigClassifierModuleCoverage(t testingT, classifier map[string]interface{}) {
	assertMapCoversStructFields(t, classifier, reflect.TypeOf(CanonicalClassifierModule{}), "global.model_catalog.modules.classifier")
	assertMapCoversStructFields(t, mustMapAt(t, classifier, "domain"), reflect.TypeOf(CanonicalCategoryModule{}), "global.model_catalog.modules.classifier.domain")
	assertMapCoversStructFields(t, mustMapAt(t, classifier, "mcp"), reflect.TypeOf(MCPCategoryModel{}), "global.model_catalog.modules.classifier.mcp")
	assertMapCoversStructFields(t, mustMapAt(t, classifier, "pii"), reflect.TypeOf(CanonicalPIIModule{}), "global.model_catalog.modules.classifier.pii")
	assertMapCoversStructFields(t, mustMapAt(t, classifier, "preference"), reflect.TypeOf(PreferenceModelConfig{}), "global.model_catalog.modules.classifier.preference")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, classifier, "preference", "prototype_scoring"),
		reflect.TypeOf(PrototypeScoringConfig{}),
		"global.model_catalog.modules.classifier.preference.prototype_scoring",
	)
}

func assertReferenceConfigComplexityModuleCoverage(t testingT, complexity map[string]interface{}) {
	assertMapCoversStructFields(t, complexity, reflect.TypeOf(ComplexityModelConfig{}), "global.model_catalog.modules.complexity")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, complexity, "prototype_scoring"),
		reflect.TypeOf(PrototypeScoringConfig{}),
		"global.model_catalog.modules.complexity.prototype_scoring",
	)
}

func assertReferenceConfigHallucinationModuleCoverage(t testingT, hallucination map[string]interface{}) {
	assertMapCoversStructFields(t, hallucination, reflect.TypeOf(CanonicalHallucinationModule{}), "global.model_catalog.modules.hallucination_mitigation")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "fact_check"), reflect.TypeOf(CanonicalFactCheckModule{}), "global.model_catalog.modules.hallucination_mitigation.fact_check")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "detector"), reflect.TypeOf(CanonicalHallucinationDetector{}), "global.model_catalog.modules.hallucination_mitigation.detector")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "explainer"), reflect.TypeOf(CanonicalExplainerModule{}), "global.model_catalog.modules.hallucination_mitigation.explainer")
}
