package config

import "reflect"

func assertSupportedSignalTypesInReferenceConfig(t testingT, root map[string]interface{}) {
	signals := mustMapAt(t, root, "routing", "signals")
	for _, signalType := range SupportedSignalTypes() {
		entry, ok := LookupSignalCatalog(signalType)
		if !ok {
			t.Fatalf("missing canonical signal catalog entry for %q", signalType)
		}
		if len(mustSliceAt(t, signals, entry.Collection)) == 0 {
			t.Fatalf("config/config.yaml must include at least one %s signal under routing.signals.%s", signalType, entry.Collection)
		}
	}
}

func assertSupportedAlgorithmsInReferenceConfig(t testingT, decisions []interface{}) {
	algorithmsByType := referenceAlgorithmsByType(t, decisions)
	for _, algorithmType := range SupportedDecisionAlgorithmTypes() {
		if _, ok := algorithmsByType[algorithmType]; !ok {
			t.Fatalf("config/config.yaml must include at least one decision.algorithm.type=%q", algorithmType)
		}
	}
	assertReferenceEscalateAlgorithmCoverage(t, decisions)
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["ratings"], "ratings"), reflect.TypeOf(RatingsAlgorithmConfig{}), "routing.decisions[].algorithm.ratings")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["remom"], "remom"), reflect.TypeOf(ReMoMAlgorithmConfig{}), "routing.decisions[].algorithm.remom")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["fusion"], "fusion"), reflect.TypeOf(FusionAlgorithmConfig{}), "routing.decisions[].algorithm.fusion")
	multiFactor := mustMapAt(t, algorithmsByType["multi_factor"], "multi_factor")
	assertMapCoversStructFields(t, multiFactor, reflect.TypeOf(MultiFactorSelectionConfig{}), "routing.decisions[].algorithm.multi_factor")
	assertMapCoversStructFields(t, mustMapAt(t, multiFactor, "quality"), reflect.TypeOf(QualityEvidenceConfig{}), "routing.decisions[].algorithm.multi_factor.quality")
}

// The Chat escalate decision covers the accept settings; the System One one
// lives in a native recipe.
func assertReferenceEscalateAlgorithmCoverage(t testingT, decisions []interface{}) {
	var escalate map[string]interface{}
	for _, algorithm := range collectChildMapsFromSlice(t, decisions, "algorithm", "routing.decisions") {
		if block, ok := algorithm["escalate"].(map[string]interface{}); ok && algorithm["type"] == DecisionAlgorithmEscalate && block["accept"] != nil {
			escalate = algorithm
		}
	}
	if escalate == nil {
		t.Fatalf("config/config.yaml must include a Chat escalate decision")
	}
	requireMapKeys(t, escalate, "routing.decisions[].algorithm(type=escalate)", "type", "budget", "on_error", "escalate")
	requireMapKeys(t, mustMapAt(t, escalate, "escalate"), "routing.decisions[].algorithm.escalate", "order", "accept")
	confidence := mustMapAt(t, escalate, "escalate", "accept", "confidence")
	assertMapCoversStructFields(t, confidence, reflect.TypeOf(EscalateConfidence{}), "routing.decisions[].algorithm.escalate.accept.confidence")
	assertMapCoversStructFields(t, mustMapAt(t, confidence, "hybrid_weights"), reflect.TypeOf(HybridWeightsConfig{}), "routing.decisions[].algorithm.escalate.accept.confidence.hybrid_weights")
}

func assertSupportedPluginsInReferenceConfig(t testingT, decisions []interface{}) {
	pluginsByType := referencePluginsByType(t, decisions)
	for _, pluginType := range SupportedDecisionPluginTypes() {
		if len(pluginsByType[pluginType]) == 0 {
			t.Fatalf("config/config.yaml must include at least one decision plugin of type %q", pluginType)
		}
	}
	assertReferenceCorePluginCoverage(t, pluginsByType)
	assertReferenceNestedPluginCoverage(t, pluginsByType)
}

func assertReferenceCorePluginCoverage(t testingT, pluginsByType map[string][]map[string]interface{}) {
	responseCachePlugins := pluginsByType["response_cache"]
	assertPluginConfigCoverage(t, responseCachePlugins, reflect.TypeOf(responseCacheReferenceConfig{}), "response_cache")
	responseCacheConfigs := collectChildMapsFromSlice(t, responseCachePlugins, "configuration", "plugins(response_cache)")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, responseCacheConfigs, "semantic", "plugins(response_cache).configuration"), reflect.TypeOf(ResponseCacheSemanticConfig{}), "plugins(response_cache).configuration.semantic")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, responseCacheConfigs, "request_controls", "plugins(response_cache).configuration"), reflect.TypeOf(ResponseCacheRequestControlsConfig{}), "plugins(response_cache).configuration.request_controls")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, responseCacheConfigs, "personalized", "plugins(response_cache).configuration"), reflect.TypeOf(ResponseCachePersonalizedConfig{}), "plugins(response_cache).configuration.personalized")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, responseCacheConfigs, "revision", "plugins(response_cache).configuration"), reflect.TypeOf(ResponseCacheRevisionConfig{}), "plugins(response_cache).configuration.revision")
	contextPlugins := pluginsByType[DecisionPluginContext]
	assertPluginConfigCoverage(t, contextPlugins, reflect.TypeOf(ContextPluginConfig{}), DecisionPluginContext)
	contextConfigs := collectChildMapsFromSlice(t, contextPlugins, "configuration", "plugins(context)")
	compressConfigs := collectChildMapsFromSlice(t, contextConfigs, "compress", "plugins(context).configuration")
	assertSliceUnionCoversStructFields(t, compressConfigs, reflect.TypeOf(ContextCompressConfig{}), "plugins(context).configuration.compress")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, contextConfigs, "budget", "plugins(context).configuration"), reflect.TypeOf(ContextBudgetConfig{}), "plugins(context).configuration.budget")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, contextConfigs, "injection", "plugins(context).configuration"), reflect.TypeOf(ContextInjectionConfig{}), "plugins(context).configuration.injection")
	compressionTargets := collectChildMapsFromSlice(t, compressConfigs, "targets", "plugins(context).configuration.compress")
	assertSliceUnionCoversStructFields(t, compressionTargets, reflect.TypeOf(ContextCompressionTargetsConfig{}), "plugins(context).configuration.compress.targets")
	targetPolicies := collectChildMapsFromSlice(t, compressionTargets, "tool_outputs", "plugins(context).configuration.compress.targets")
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "history", "plugins(context).configuration.compress.targets")...)
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "rag", "plugins(context).configuration.compress.targets")...)
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "memory", "plugins(context).configuration.compress.targets")...)
	assertSliceUnionCoversStructFields(t, targetPolicies, reflect.TypeOf(ContextCompressionTargetConfig{}), "plugins(context).configuration.compress.targets.*")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressConfigs, "recovery", "plugins(context).configuration.compress"), reflect.TypeOf(ContextCompressionRecoveryConfig{}), "plugins(context).configuration.compress.recovery")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressConfigs, "request_controls", "plugins(context).configuration.compress"), reflect.TypeOf(ContextCompressionRequestControlsConfig{}), "plugins(context).configuration.compress.request_controls")
	assertPluginConfigCoverage(t, pluginsByType["fast_response"], reflect.TypeOf(FastResponsePluginConfig{}), "fast_response")
	assertPluginConfigCoverage(t, pluginsByType["system_prompt"], reflect.TypeOf(SystemPromptPluginConfig{}), "system_prompt")
	assertPluginConfigCoverage(t, pluginsByType["header_mutation"], reflect.TypeOf(HeaderMutationPluginConfig{}), "header_mutation")
	assertPluginConfigCoverage(t, pluginsByType["response_guard"], reflect.TypeOf(ResponseGuardPluginConfig{}), "response_guard")
	guardConfigs := collectChildMapsFromSlice(t, pluginsByType["response_guard"], "configuration", "plugins(response_guard)")
	assertSliceUnionCoversStructFields(t, collectNestedSliceItems(t, guardConfigs, "rules", "plugins(response_guard).configuration"), reflect.TypeOf(ResponseGuardRule{}), "plugins(response_guard).configuration.rules")
	assertPluginConfigCoverage(t, pluginsByType["masking"], reflect.TypeOf(MaskingPluginConfig{}), "masking")
	assertPluginConfigCoverage(t, pluginsByType["router_replay"], reflect.TypeOf(RouterReplayPluginConfig{}), "router_replay")
	assertPluginConfigCoverage(t, pluginsByType["request_params"], reflect.TypeOf(RequestParamsPluginConfig{}), "request_params")
	assertPluginConfigCoverage(t, pluginsByType["tools"], reflect.TypeOf(ToolsPluginConfig{}), "tools")
	assertPluginConfigCoverage(t, pluginsByType["prompt_cache"], reflect.TypeOf(PromptCachePluginConfig{}), "prompt_cache")
}

type responseCacheReferenceConfig struct {
	Enabled         bool                                `yaml:"enabled"`
	Mode            string                              `yaml:"mode,omitempty"`
	Scope           string                              `yaml:"scope,omitempty"`
	Semantic        *ResponseCacheSemanticConfig        `yaml:"semantic,omitempty"`
	RequestControls *ResponseCacheRequestControlsConfig `yaml:"request_controls,omitempty"`
	Personalized    *ResponseCachePersonalizedConfig    `yaml:"personalized,omitempty"`
	Revision        *ResponseCacheRevisionConfig        `yaml:"revision,omitempty"`
	TTLSeconds      *int                                `yaml:"ttl_seconds,omitempty"`
}

func assertReferenceNestedPluginCoverage(t testingT, pluginsByType map[string][]map[string]interface{}) {
	assertReferenceContextSourceCoverage(t, pluginsByType[DecisionPluginContext])
	assertReferenceHeaderMutationCoverage(t, pluginsByType["header_mutation"])
}

// assertReferenceContextSourceCoverage checks that the reference context
// plugins' sources write every source field, every RAG backend's
// configuration, the memory reflection and the match block.
func assertReferenceContextSourceCoverage(t testingT, plugins []map[string]interface{}) {
	configs := collectChildMapsFromSlice(t, plugins, "configuration", "plugins(context)")
	sources := collectNestedSliceItems(t, configs, "sources", "plugins(context).configuration")
	assertSliceUnionCoversStructFields(t, sources, reflect.TypeOf(ContextSource{}), "routing.decisions[].plugins[type=context].configuration.sources")
	var ragSources, memorySources []map[string]interface{}
	for _, item := range sources {
		source, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("a context source is %T, not a mapping", item)
		}
		if source["type"] == ContextSourceMemory {
			memorySources = append(memorySources, source)
		} else {
			ragSources = append(ragSources, source)
		}
	}
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, memorySources, "reflection", "sources(memory)"), reflect.TypeOf(MemoryReflectionConfig{}), "sources[type=memory].reflection")
	matches := collectChildMapsFromSlice(t, ragSources, "match", "sources(rag)")
	assertSliceUnionCoversStructFields(t, matches, reflect.TypeOf(MatchConfig{}), "sources[type=rag].match")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, matches, "recall", "sources(rag).match"), reflect.TypeOf(MatchRecall{}), "sources[type=rag].match.recall")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, matches, "judge", "sources(rag).match"), reflect.TypeOf(MatchJudge{}), "sources[type=rag].match.judge", "question")
	ragByBackend := mapByStringField(t, ragSources, "backend", "rag")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["milvus"], "backend_config"), reflect.TypeOf(MilvusRAGConfig{}), "rag.backend_config(milvus)")
	// The reference's custom request format searches without a query vector, so it takes no embedding.
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["external_api"], "backend_config"), reflect.TypeOf(ExternalAPIRAGConfig{}), "rag.backend_config(external_api)", "embedding")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["mcp"], "backend_config"), reflect.TypeOf(MCPRAGConfig{}), "rag.backend_config(mcp)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["openai"], "backend_config"), reflect.TypeOf(OpenAIRAGConfig{}), "rag.backend_config(openai)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["vectorstore"], "backend_config"), reflect.TypeOf(VectorStoreRAGConfig{}), "rag.backend_config(vectorstore)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["hybrid"], "backend_config"), reflect.TypeOf(HybridRAGConfig{}), "rag.backend_config(hybrid)")
}

func assertReferenceHeaderMutationCoverage(t testingT, plugins []map[string]interface{}) {
	configs := collectChildMapsFromSlice(t, plugins, "configuration", "plugins(header_mutation)")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, configs, "add", "plugins(header_mutation).configuration"),
		reflect.TypeOf(HeaderPair{}),
		"routing.decisions[].plugins[type=header_mutation].configuration.add",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, configs, "update", "plugins(header_mutation).configuration"),
		reflect.TypeOf(HeaderPair{}),
		"routing.decisions[].plugins[type=header_mutation].configuration.update",
	)
}

func assertDecisionRuleCompositionInReferenceConfig(t testingT, decisions []interface{}) {
	ruleOperators := make(map[string]bool)
	ruleTypes := make(map[string]bool)

	for _, rawDecision := range decisions {
		decision := mustMapValue(t, rawDecision, "routing.decisions")
		collectRuleCoverage(t, mustMapAt(t, decision, "rules"), ruleOperators, ruleTypes)
	}
	for _, operator := range []string{"AND", "OR", "NOT"} {
		if !ruleOperators[operator] {
			t.Fatalf("config/config.yaml must exercise %s rule composition in routing.decisions", operator)
		}
	}
}

func assertReferenceLoRACatalogCoverage(t testingT, root map[string]interface{}) {
	modelCards := mustSliceAt(t, root, "routing", "modelCards")
	if len(collectNestedSliceItems(t, modelCards, "loras", "routing.modelCards")) == 0 {
		t.Fatalf("config/config.yaml must declare at least one routing.modelCards[].loras entry")
	}

	decisions := mustSliceAt(t, root, "routing", "decisions")
	modelRefs := collectNestedSliceItems(t, decisions, "modelRefs", "routing.decisions")
	for _, rawModelRef := range modelRefs {
		modelRef := mustMapValue(t, rawModelRef, "routing.decisions[].modelRefs")
		if _, ok := modelRef["lora_name"]; ok {
			return
		}
	}
	t.Fatalf("config/config.yaml must exercise routing.decisions[].modelRefs[].lora_name")
}

func referenceAlgorithmsByType(t testingT, decisions []interface{}) map[string]map[string]interface{} {
	algorithms := collectChildMapsFromSlice(t, decisions, "algorithm", "routing.decisions")
	result := make(map[string]map[string]interface{}, len(algorithms))
	for _, rawAlgorithm := range algorithms {
		algorithmType := mustStringAt(t, rawAlgorithm, "type")
		result[algorithmType] = rawAlgorithm
	}
	return result
}

func referencePluginsByType(t testingT, decisions []interface{}) map[string][]map[string]interface{} {
	result := make(map[string][]map[string]interface{})
	for _, rawPlugin := range collectNestedSliceItems(t, decisions, "plugins", "routing.decisions") {
		plugin := mustMapValue(t, rawPlugin, "routing.decisions[].plugins")
		pluginType := mustStringAt(t, plugin, "type")
		result[pluginType] = append(result[pluginType], plugin)
	}
	return result
}
