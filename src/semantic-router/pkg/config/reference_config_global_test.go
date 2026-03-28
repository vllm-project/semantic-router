package config

import "reflect"

func assertReferenceConfigRouterGlobalCoverage(t testingT, router map[string]interface{}) {
	modelSelection := mustMapAt(t, router, "model_selection")

	assertMapCoversStructFields(t, router, reflect.TypeOf(CanonicalRouterGlobal{}), "global.router")
	assertMapCoversStructFields(t, mustMapAt(t, router, "streamed_body"), reflect.TypeOf(CanonicalStreamedBody{}), "global.router.streamed_body")
	assertMapCoversStructFields(t, modelSelection, reflect.TypeOf(ModelSelectionConfig{}), "global.router.model_selection")
	assertReferenceConfigRouterSelectionCoverage(t, modelSelection)
}

func assertReferenceConfigRouterSelectionCoverage(t testingT, modelSelection map[string]interface{}) {
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "ml"), reflect.TypeOf(MLSelectionConfig{}), "global.router.model_selection.ml")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "elo"), reflect.TypeOf(EloSelectionConfig{}), "global.router.model_selection.elo")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "router_dc"), reflect.TypeOf(RouterDCSelectionConfig{}), "global.router.model_selection.router_dc")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "automix"), reflect.TypeOf(AutoMixSelectionConfig{}), "global.router.model_selection.automix")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "hybrid"), reflect.TypeOf(HybridSelectionConfig{}), "global.router.model_selection.hybrid")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "ml", "knn"), reflect.TypeOf(MLKNNConfig{}), "global.router.model_selection.ml.knn")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "ml", "kmeans"), reflect.TypeOf(MLKMeansConfig{}), "global.router.model_selection.ml.kmeans")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "ml", "svm"), reflect.TypeOf(MLSVMConfig{}), "global.router.model_selection.ml.svm")
	assertMapCoversStructFields(t, mustMapAt(t, modelSelection, "ml", "mlp"), reflect.TypeOf(MLMLPConfig{}), "global.router.model_selection.ml.mlp")
}

func assertReferenceConfigServiceGlobalCoverage(t testingT, services map[string]interface{}) {
	assertMapCoversStructFields(t, services, reflect.TypeOf(CanonicalServiceGlobal{}), "global.services")
	assertReferenceConfigAPIServiceCoverage(t, mustMapAt(t, services, "api"))
	assertReferenceConfigResponseAPIServiceCoverage(t, mustMapAt(t, services, "response_api"))
	assertReferenceConfigObservabilityCoverage(t, mustMapAt(t, services, "observability"))
	assertReferenceConfigAuthzCoverage(t, mustMapAt(t, services, "authz"))
	assertReferenceConfigRateLimitCoverage(t, mustMapAt(t, services, "ratelimit"))
	assertReferenceConfigRouterReplayCoverage(t, mustMapAt(t, services, "router_replay"))
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
}

func assertReferenceConfigStoreGlobalCoverage(t testingT, stores map[string]interface{}) {
	assertMapCoversStructFields(t, stores, reflect.TypeOf(CanonicalStoreGlobal{}), "global.stores")
	assertReferenceConfigSemanticCacheCoverage(t, mustMapAt(t, stores, "semantic_cache"))
	assertReferenceConfigMemoryCoverage(t, mustMapAt(t, stores, "memory"))
	assertReferenceConfigVectorStoreCoverage(t, mustMapAt(t, stores, "vector_store"))
}

func assertReferenceConfigSemanticCacheCoverage(t testingT, semanticCache map[string]interface{}) {
	assertMapCoversStructFields(t, semanticCache, reflect.TypeOf(SemanticCache{}), "global.stores.semantic_cache")
	assertMapCoversStructFields(t, mustMapAt(t, semanticCache, "redis"), reflect.TypeOf(RedisConfig{}), "global.stores.semantic_cache.redis")
	assertMapCoversStructFields(t, mustMapAt(t, semanticCache, "valkey"), reflect.TypeOf(ValkeyConfig{}), "global.stores.semantic_cache.valkey")
	assertMapCoversStructFields(t, mustMapAt(t, semanticCache, "milvus"), reflect.TypeOf(MilvusConfig{}), "global.stores.semantic_cache.milvus")
}

func assertReferenceConfigMemoryCoverage(t testingT, memory map[string]interface{}) {
	assertMapCoversStructFields(t, memory, reflect.TypeOf(MemoryConfig{}), "global.stores.memory")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "milvus"), reflect.TypeOf(MemoryMilvusConfig{}), "global.stores.memory.milvus")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "quality_scoring"), reflect.TypeOf(MemoryQualityScoringConfig{}), "global.stores.memory.quality_scoring")
	assertMapCoversStructFields(t, mustMapAt(t, memory, "reflection"), reflect.TypeOf(MemoryReflectionConfig{}), "global.stores.memory.reflection")
}

func assertReferenceConfigVectorStoreCoverage(t testingT, vectorStore map[string]interface{}) {
	assertMapCoversStructFields(t, vectorStore, reflect.TypeOf(VectorStoreConfig{}), "global.stores.vector_store")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "memory"), reflect.TypeOf(VectorStoreMemoryConfig{}), "global.stores.vector_store.memory")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "llama_stack"), reflect.TypeOf(LlamaStackVectorStoreConfig{}), "global.stores.vector_store.llama_stack")
	assertMapCoversStructFields(t, mustMapAt(t, vectorStore, "milvus"), reflect.TypeOf(MilvusConfig{}), "global.stores.vector_store.milvus")
}

func assertReferenceConfigIntegrationGlobalCoverage(t testingT, integrations map[string]interface{}) {
	tools := mustMapAt(t, integrations, "tools")

	assertMapCoversStructFields(t, integrations, reflect.TypeOf(CanonicalIntegrationGlobal{}), "global.integrations")
	assertMapCoversStructFields(t, tools, reflect.TypeOf(ToolsConfig{}), "global.integrations.tools")
	assertMapCoversStructFields(t, mustMapAt(t, tools, "advanced_filtering"), reflect.TypeOf(AdvancedToolFilteringConfig{}), "global.integrations.tools.advanced_filtering")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, tools, "advanced_filtering", "weights"),
		reflect.TypeOf(ToolFilteringWeights{}),
		"global.integrations.tools.advanced_filtering.weights",
	)
	assertMapCoversStructFields(t, mustMapAt(t, integrations, "looper"), reflect.TypeOf(LooperConfig{}), "global.integrations.looper")
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
}

func assertReferenceConfigModelModuleCoverage(t testingT, modules map[string]interface{}) {
	assertMapCoversStructFields(t, modules, reflect.TypeOf(CanonicalModelModules{}), "global.model_catalog.modules")
	assertMapCoversStructFields(t, mustMapAt(t, modules, "prompt_compression"), reflect.TypeOf(PromptCompressionConfig{}), "global.model_catalog.modules.prompt_compression")
	assertMapCoversStructFields(t, mustMapAt(t, modules, "prompt_guard"), reflect.TypeOf(CanonicalPromptGuardModule{}), "global.model_catalog.modules.prompt_guard")
	assertReferenceConfigClassifierModuleCoverage(t, mustMapAt(t, modules, "classifier"))
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
}

func assertReferenceConfigHallucinationModuleCoverage(t testingT, hallucination map[string]interface{}) {
	assertMapCoversStructFields(t, hallucination, reflect.TypeOf(CanonicalHallucinationModule{}), "global.model_catalog.modules.hallucination_mitigation")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "fact_check"), reflect.TypeOf(CanonicalFactCheckModule{}), "global.model_catalog.modules.hallucination_mitigation.fact_check")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "detector"), reflect.TypeOf(CanonicalHallucinationDetector{}), "global.model_catalog.modules.hallucination_mitigation.detector")
	assertMapCoversStructFields(t, mustMapAt(t, hallucination, "explainer"), reflect.TypeOf(CanonicalExplainerModule{}), "global.model_catalog.modules.hallucination_mitigation.explainer")
}
