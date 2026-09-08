package config

import "reflect"

var referenceSignalKeyByType = map[string]string{
	SignalTypeAuthz:         "role_bindings",
	SignalTypeComplexity:    "complexity",
	SignalTypeContext:       "context",
	SignalTypeDomain:        "domains",
	SignalTypeEmbedding:     "embeddings",
	SignalTypeFactCheck:     "fact_check",
	SignalTypeJailbreak:     "jailbreak",
	SignalTypeKeyword:       "keywords",
	SignalTypeLanguage:      "language",
	SignalTypeModality:      "modality",
	SignalTypePII:           "pii",
	SignalTypePreference:    "preferences",
	SignalTypeReask:         "reasks",
	SignalTypeStructure:     "structure",
	SignalTypeConversation:  "conversation",
	SignalTypeKB:            "kb",
	SignalTypeUserFeedback:  "user_feedbacks",
	SignalTypeEvent:         "events",
	SignalTypeMetadata:      "metadata",
	SignalTypeClassifier:    "classifiers",
	SignalTypeInputModality: "input_modality",
}

func assertSupportedSignalTypesInReferenceConfig(t testingT, root map[string]interface{}) {
	signals := mustMapAt(t, root, "routing", "signals")
	for _, signalType := range SupportedSignalTypes() {
		key, ok := referenceSignalKeyByType[signalType]
		if !ok {
			t.Fatalf("missing canonical signal key mapping for %q", signalType)
		}
		if len(mustSliceAt(t, signals, key)) == 0 {
			t.Fatalf("config/config.yaml must include at least one %s signal under routing.signals.%s", signalType, key)
		}
	}

	coveredClassifierTypes := make(map[string]bool)
	for _, rawClassifier := range mustSliceAt(t, signals, "classifiers") {
		classifier := mustMapValue(t, rawClassifier, "routing.signals.classifiers")
		coveredClassifierTypes[mustStringAt(t, classifier, "type")] = true
	}
	for _, classifierType := range []string{
		ClassifierSignalTypeLocal,
		ClassifierSignalTypeLLM,
		ClassifierSignalTypeSequenceClassifier,
	} {
		if !coveredClassifierTypes[classifierType] {
			t.Fatalf("config/config.yaml must include classifier signal type %q", classifierType)
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
	assertReferenceConfidenceAlgorithmCoverage(t, algorithmsByType)
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["ratings"], "ratings"), reflect.TypeOf(RatingsAlgorithmConfig{}), "routing.decisions[].algorithm.ratings")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["remom"], "remom"), reflect.TypeOf(ReMoMAlgorithmConfig{}), "routing.decisions[].algorithm.remom")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["fusion"], "fusion"), reflect.TypeOf(FusionAlgorithmConfig{}), "routing.decisions[].algorithm.fusion")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["router_dc"], "router_dc"), reflect.TypeOf(RouterDCSelectionConfig{}), "routing.decisions[].algorithm.router_dc")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["automix"], "automix"), reflect.TypeOf(AutoMixSelectionConfig{}), "routing.decisions[].algorithm.automix")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["hybrid"], "hybrid"), reflect.TypeOf(HybridSelectionConfig{}), "routing.decisions[].algorithm.hybrid")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["latency_aware"], "latency_aware"), reflect.TypeOf(LatencyAwareAlgorithmConfig{}), "routing.decisions[].algorithm.latency_aware")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["multi_factor"], "multi_factor"), reflect.TypeOf(MultiFactorSelectionConfig{}), "routing.decisions[].algorithm.multi_factor")
	assertMapCoversStructFields(t, mustMapAt(t, algorithmsByType["prompt"], "prompt"), reflect.TypeOf(PromptSelectionConfig{}), "routing.decisions[].algorithm.prompt")
}

func assertReferenceConfidenceAlgorithmCoverage(t testingT, algorithmsByType map[string]map[string]interface{}) {
	confidence := algorithmsByType["confidence"]
	requireMapKeys(t, confidence, "routing.decisions[].algorithm(type=confidence)", "type", "confidence", "on_error")
	assertMapCoversStructFields(t, mustMapAt(t, confidence, "confidence"), reflect.TypeOf(ConfidenceAlgorithmConfig{}), "routing.decisions[].algorithm.confidence")
	assertMapCoversStructFields(
		t,
		mustMapAt(t, confidence, "confidence", "hybrid_weights"),
		reflect.TypeOf(HybridWeightsConfig{}),
		"routing.decisions[].algorithm.confidence.hybrid_weights",
	)
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
	compressionPlugins := pluginsByType["context_compression"]
	assertPluginConfigCoverage(t, compressionPlugins, reflect.TypeOf(contextCompressionReferenceConfig{}), "context_compression")
	compressionConfigs := collectChildMapsFromSlice(t, compressionPlugins, "configuration", "plugins(context_compression)")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressionConfigs, "budget", "plugins(context_compression).configuration"), reflect.TypeOf(ContextCompressionBudgetConfig{}), "plugins(context_compression).configuration.budget")
	compressionTargets := collectChildMapsFromSlice(t, compressionConfigs, "targets", "plugins(context_compression).configuration")
	assertSliceUnionCoversStructFields(t, compressionTargets, reflect.TypeOf(ContextCompressionTargetsConfig{}), "plugins(context_compression).configuration.targets")
	targetPolicies := collectChildMapsFromSlice(t, compressionTargets, "tool_outputs", "plugins(context_compression).configuration.targets")
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "history", "plugins(context_compression).configuration.targets")...)
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "rag", "plugins(context_compression).configuration.targets")...)
	targetPolicies = append(targetPolicies, collectChildMapsFromSlice(t, compressionTargets, "memory", "plugins(context_compression).configuration.targets")...)
	assertSliceUnionCoversStructFields(t, targetPolicies, reflect.TypeOf(ContextCompressionTargetConfig{}), "plugins(context_compression).configuration.targets.*")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressionConfigs, "scoring", "plugins(context_compression).configuration"), reflect.TypeOf(ContextCompressionScoringConfig{}), "plugins(context_compression).configuration.scoring")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressionConfigs, "recovery", "plugins(context_compression).configuration"), reflect.TypeOf(ContextCompressionRecoveryConfig{}), "plugins(context_compression).configuration.recovery")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, compressionConfigs, "request_controls", "plugins(context_compression).configuration"), reflect.TypeOf(ContextCompressionRequestControlsConfig{}), "plugins(context_compression).configuration.request_controls")
	assertPluginConfigCoverage(t, pluginsByType["memory"], reflect.TypeOf(MemoryPluginConfig{}), "memory")
	assertPluginConfigCoverage(t, pluginsByType["fast_response"], reflect.TypeOf(FastResponsePluginConfig{}), "fast_response")
	assertPluginConfigCoverage(t, pluginsByType["system_prompt"], reflect.TypeOf(SystemPromptPluginConfig{}), "system_prompt")
	assertPluginConfigCoverage(t, pluginsByType["header_mutation"], reflect.TypeOf(HeaderMutationPluginConfig{}), "header_mutation")
	assertPluginConfigCoverage(t, pluginsByType["hallucination"], reflect.TypeOf(HallucinationPluginConfig{}), "hallucination")
	assertPluginConfigCoverage(t, pluginsByType["response_jailbreak"], reflect.TypeOf(ResponseJailbreakPluginConfig{}), "response_jailbreak")
	assertPluginConfigCoverage(t, pluginsByType["router_replay"], reflect.TypeOf(RouterReplayPluginConfig{}), "router_replay")
	assertPluginConfigCoverage(t, pluginsByType["request_params"], reflect.TypeOf(RequestParamsPluginConfig{}), "request_params")
	assertPluginConfigCoverage(t, pluginsByType["tool_selection"], reflect.TypeOf(ToolSelectionPluginConfig{}), "tool_selection")
}

type contextCompressionReferenceConfig struct {
	Enabled         bool                                     `yaml:"enabled"`
	Mode            string                                   `yaml:"mode,omitempty"`
	Budget          *ContextCompressionBudgetConfig          `yaml:"budget,omitempty"`
	Targets         *ContextCompressionTargetsConfig         `yaml:"targets,omitempty"`
	Scoring         *ContextCompressionScoringConfig         `yaml:"scoring,omitempty"`
	Recovery        *ContextCompressionRecoveryConfig        `yaml:"recovery,omitempty"`
	RequestControls *ContextCompressionRequestControlsConfig `yaml:"request_controls,omitempty"`
	FailureMode     string                                   `yaml:"failure_mode,omitempty"`
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
	assertReferenceMemoryPluginCoverage(t, pluginsByType["memory"])
	assertReferenceHeaderMutationCoverage(t, pluginsByType["header_mutation"])
	assertReferenceRAGPluginCoverage(t, pluginsByType["rag"])
}

func assertReferenceMemoryPluginCoverage(t testingT, plugins []map[string]interface{}) {
	memoryConfigs := collectChildMapsFromSlice(t, plugins, "configuration", "plugins(memory)")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, memoryConfigs, "reflection", "plugins(memory).configuration"),
		reflect.TypeOf(MemoryReflectionConfig{}),
		"routing.decisions[].plugins[type=memory].configuration.reflection",
	)
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

func assertReferenceRAGPluginCoverage(t testingT, plugins []map[string]interface{}) {
	configs := collectChildMapsFromSlice(t, plugins, "configuration", "plugins(rag)")
	ragByBackend := mapByStringField(t, configs, "backend", "rag")

	assertSliceUnionCoversStructFields(t, configs, reflect.TypeOf(RAGPluginConfig{}), "routing.decisions[].plugins[type=rag].configuration")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["milvus"], "backend_config"), reflect.TypeOf(MilvusRAGConfig{}), "rag.backend_config(milvus)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["external_api"], "backend_config"), reflect.TypeOf(ExternalAPIRAGConfig{}), "rag.backend_config(external_api)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["mcp"], "backend_config"), reflect.TypeOf(MCPRAGConfig{}), "rag.backend_config(mcp)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["openai"], "backend_config"), reflect.TypeOf(OpenAIRAGConfig{}), "rag.backend_config(openai)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["vectorstore"], "backend_config"), reflect.TypeOf(VectorStoreRAGConfig{}), "rag.backend_config(vectorstore)")
	assertMapCoversStructFields(t, mustMapAt(t, ragByBackend["hybrid"], "backend_config"), reflect.TypeOf(HybridRAGConfig{}), "rag.backend_config(hybrid)")
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
