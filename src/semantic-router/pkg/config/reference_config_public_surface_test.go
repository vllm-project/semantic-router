package config

import "reflect"

func assertReferenceConfigTopLevelCoverage(t testingT, root map[string]interface{}) {
	assertMapCoversStructFields(t, root, reflect.TypeOf(CanonicalConfig{}), "config")

	listeners := mustSliceAt(t, root, "listeners")
	assertSliceUnionCoversStructFields(t, listeners, reflect.TypeOf(Listener{}), "listeners")
}

func assertReferenceConfigProviderCoverage(t testingT, root map[string]interface{}) {
	providers := mustMapAt(t, root, "providers")
	defaults := mustMapAt(t, providers, "defaults")
	reasoningFamilies := mustMapAt(t, defaults, "reasoning_families")
	models := mustSliceAt(t, providers, "models")

	assertMapCoversStructFields(t, providers, reflect.TypeOf(CanonicalProviders{}), "providers")
	assertMapCoversStructFields(t, defaults, reflect.TypeOf(CanonicalProviderDefaults{}), "providers.defaults")
	assertSliceUnionCoversStructFields(
		t,
		mapValuesToSlice(t, reasoningFamilies, "providers.defaults.reasoning_families"),
		reflect.TypeOf(ReasoningFamilyConfig{}),
		"providers.defaults.reasoning_families",
	)
	assertSliceUnionCoversStructFields(t, models, reflect.TypeOf(CanonicalProviderModel{}), "providers.models")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, models, "backend_refs", "providers.models"),
		reflect.TypeOf(CanonicalBackendRef{}),
		"providers.models[].backend_refs",
	)
}

func assertReferenceConfigRoutingCoverage(t testingT, root map[string]interface{}) {
	routing := mustMapAt(t, root, "routing")

	assertMapCoversStructFields(t, routing, reflect.TypeOf(CanonicalRouting{}), "routing")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, routing, "modelCards"),
		reflect.TypeOf(RoutingModel{}),
		"routing.modelCards",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, routing, "modelCards"), "loras", "routing.modelCards"),
		reflect.TypeOf(LoRAAdapter{}),
		"routing.modelCards[].loras",
	)
	assertReferenceConfigSignalCoverage(t, mustMapAt(t, routing, "signals"))
	assertReferenceConfigProjectionCoverage(t, mustMapAt(t, routing, "projections"))
	assertReferenceConfigDecisionCoverage(t, mustSliceAt(t, routing, "decisions"))
	assertReferenceConfigSessionStateCoverage(t, mustSliceAt(t, routing, "session_states"))
}

func assertReferenceConfigSessionStateCoverage(t testingT, sessionStates []interface{}) {
	assertSliceUnionCoversStructFields(t, sessionStates, reflect.TypeOf(SessionStateConfig{}), "routing.session_states")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, sessionStates, "fields", "routing.session_states"),
		reflect.TypeOf(SessionStateFieldConfig{}),
		"routing.session_states[].fields",
	)
}

func assertReferenceConfigSignalCoverage(t testingT, signals map[string]interface{}) {
	assertMapCoversStructFields(t, signals, reflect.TypeOf(CanonicalSignals{}), "routing.signals")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "keywords"), reflect.TypeOf(KeywordRule{}), "routing.signals.keywords")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "embeddings"), reflect.TypeOf(EmbeddingRule{}), "routing.signals.embeddings")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "domains"), reflect.TypeOf(Category{}), "routing.signals.domains")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, signals, "domains"), "model_scores", "routing.signals.domains"),
		reflect.TypeOf(ModelScore{}),
		"routing.signals.domains[].model_scores",
	)
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "fact_check"), reflect.TypeOf(FactCheckRule{}), "routing.signals.fact_check")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "user_feedbacks"), reflect.TypeOf(UserFeedbackRule{}), "routing.signals.user_feedbacks")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "reasks"), reflect.TypeOf(ReaskRule{}), "routing.signals.reasks")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "preferences"), reflect.TypeOf(PreferenceRule{}), "routing.signals.preferences")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "language"), reflect.TypeOf(LanguageRule{}), "routing.signals.language")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "context"), reflect.TypeOf(ContextRule{}), "routing.signals.context")
	assertReferenceConfigStructureCoverage(t, mustSliceAt(t, signals, "structure"))
	assertReferenceConfigComplexityCoverage(t, mustSliceAt(t, signals, "complexity"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "modality"), reflect.TypeOf(ModalityRule{}), "routing.signals.modality")
	assertReferenceConfigRoleBindingCoverage(t, mustSliceAt(t, signals, "role_bindings"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "jailbreak"), reflect.TypeOf(JailbreakRule{}), "routing.signals.jailbreak")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "pii"), reflect.TypeOf(PIIRule{}), "routing.signals.pii")
	assertReferenceConfigKBSignalCoverage(t, mustSliceAt(t, signals, "kb"))
}

func assertReferenceConfigProjectionCoverage(t testingT, projections map[string]interface{}) {
	assertMapCoversStructFields(t, projections, reflect.TypeOf(CanonicalProjections{}), "routing.projections")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "partitions"),
		reflect.TypeOf(ProjectionPartition{}),
		"routing.projections.partitions",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "scores"),
		reflect.TypeOf(ProjectionScore{}),
		"routing.projections.scores",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, projections, "scores"), "inputs", "routing.projections.scores"),
		reflect.TypeOf(ProjectionScoreInput{}),
		"routing.projections.scores[].inputs",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "mappings"),
		reflect.TypeOf(ProjectionMapping{}),
		"routing.projections.mappings",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, mustSliceAt(t, projections, "mappings"), "calibration", "routing.projections.mappings"),
		reflect.TypeOf(ProjectionMappingCalibration{}),
		"routing.projections.mappings[].calibration",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, projections, "mappings"), "outputs", "routing.projections.mappings"),
		reflect.TypeOf(ProjectionMappingOutput{}),
		"routing.projections.mappings[].outputs",
	)
}

func assertReferenceConfigComplexityCoverage(t testingT, complexity []interface{}) {
	assertSliceUnionCoversStructFields(t, complexity, reflect.TypeOf(ComplexityRule{}), "routing.signals.complexity")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, complexity, "hard", "routing.signals.complexity"),
		reflect.TypeOf(ComplexityCandidates{}),
		"routing.signals.complexity[].hard",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, complexity, "easy", "routing.signals.complexity"),
		reflect.TypeOf(ComplexityCandidates{}),
		"routing.signals.complexity[].easy",
	)
}

func assertReferenceConfigStructureCoverage(t testingT, structure []interface{}) {
	assertSliceUnionCoversStructFields(t, structure, reflect.TypeOf(StructureRule{}), "routing.signals.structure")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, structure, "feature", "routing.signals.structure"),
		reflect.TypeOf(StructureFeature{}),
		"routing.signals.structure[].feature",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, structure, "predicate", "routing.signals.structure"),
		reflect.TypeOf(NumericPredicate{}),
		"routing.signals.structure[].predicate",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(
			t,
			collectChildMapsFromSlice(t, structure, "feature", "routing.signals.structure"),
			"source",
			"routing.signals.structure[].feature",
		),
		reflect.TypeOf(StructureSource{}),
		"routing.signals.structure[].feature.source",
	)
}

func assertReferenceConfigRoleBindingCoverage(t testingT, roleBindings []interface{}) {
	assertSliceUnionCoversStructFields(t, roleBindings, reflect.TypeOf(RoleBinding{}), "routing.signals.role_bindings")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, roleBindings, "subjects", "routing.signals.role_bindings"),
		reflect.TypeOf(Subject{}),
		"routing.signals.role_bindings[].subjects",
	)
}

func assertReferenceConfigKBSignalCoverage(t testingT, kb []interface{}) {
	assertSliceUnionCoversStructFields(t, kb, reflect.TypeOf(KBSignalRule{}), "routing.signals.kb")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, kb, "target", "routing.signals.kb"),
		reflect.TypeOf(KBSignalTarget{}),
		"routing.signals.kb[].target",
	)
}

func assertReferenceConfigDecisionCoverage(t testingT, decisions []interface{}) {
	assertSliceUnionCoversStructFields(t, decisions, reflect.TypeOf(Decision{}), "routing.decisions")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, decisions, "modelRefs", "routing.decisions"),
		reflect.TypeOf(ModelRef{}),
		"routing.decisions[].modelRefs",
	)
	candidateIterations := collectNestedSliceItems(t, decisions, "candidateIterations", "routing.decisions")
	assertSliceUnionCoversStructFields(
		t,
		candidateIterations,
		reflect.TypeOf(CandidateIterationConfig{}),
		"routing.decisions[].candidateIterations",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, candidateIterations, "models", "routing.decisions[].candidateIterations"),
		reflect.TypeOf(ModelRef{}),
		"routing.decisions[].candidateIterations[].models",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, candidateIterations, "outputs", "routing.decisions[].candidateIterations"),
		reflect.TypeOf(CandidateIterationOutputConfig{}),
		"routing.decisions[].candidateIterations[].outputs",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, decisions, "algorithm", "routing.decisions"),
		reflect.TypeOf(AlgorithmConfig{}),
		"routing.decisions[].algorithm",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, decisions, "plugins", "routing.decisions"),
		reflect.TypeOf(DecisionPlugin{}),
		"routing.decisions[].plugins",
	)
}

func assertReferenceConfigGlobalCoverage(t testingT, root map[string]interface{}) {
	global := mustMapAt(t, root, "global")

	assertMapCoversStructFields(t, global, reflect.TypeOf(CanonicalGlobal{}), "global")
	assertReferenceConfigRouterGlobalCoverage(t, mustMapAt(t, global, "router"))
	assertReferenceConfigServiceGlobalCoverage(t, mustMapAt(t, global, "services"))
	assertReferenceConfigStoreGlobalCoverage(t, mustMapAt(t, global, "stores"))
	assertReferenceConfigIntegrationGlobalCoverage(t, mustMapAt(t, global, "integrations"))
	assertReferenceConfigModelCatalogCoverage(t, mustMapAt(t, global, "model_catalog"))
}
