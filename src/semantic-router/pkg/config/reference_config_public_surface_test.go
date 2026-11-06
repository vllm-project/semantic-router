package config

import (
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func assertReferenceConfigTopLevelCoverage(t testingT, root map[string]interface{}) {
	assertMapCoversStructFields(t, root, reflect.TypeOf(CanonicalConfig{}), "config")

	listeners := mustSliceAt(t, root, "listeners")
	assertSliceUnionCoversStructFields(t, listeners, reflect.TypeOf(Listener{}), "listeners")
}

func assertReferenceConfigProviderCoverage(t testingT, root map[string]interface{}) {
	providers := mustMapAt(t, root, "providers")
	defaults := mustMapAt(t, providers, "defaults")
	models := mustSliceAt(t, providers, "models")

	assertMapCoversStructFields(t, providers, reflect.TypeOf(CanonicalProviders{}), "providers")
	assertMapCoversStructFields(t, defaults, reflect.TypeOf(CanonicalProviderDefaults{}), "providers.defaults")
	assertSliceUnionCoversStructFields(t, models, reflect.TypeOf(CanonicalProviderModel{}), "providers.models")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, models, "reasoning", "providers.models"),
		reflect.TypeOf(CanonicalReasoning{}),
		"providers.models[].reasoning",
	)
	backendRefs := collectNestedSliceItems(t, models, "backend_refs", "providers.models")
	// base_url, api_key, api_key_env, auth_header, auth_prefix and
	// extra_headers are the deprecated spellings of endpoint, auth and headers.
	assertSliceUnionCoversStructFields(
		t,
		backendRefs,
		reflect.TypeOf(CanonicalBackendRef{}),
		"providers.models[].backend_refs",
		"base_url", "api_key", "api_key_env", "auth_header", "auth_prefix", "extra_headers",
	)
	// A backend reads its key from api_key_env only.
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, backendRefs, "auth", "providers.models[].backend_refs"),
		reflect.TypeOf(DeploymentAuth{}),
		"providers.models[].backend_refs[].auth",
		"api_key_file",
	)
}

func assertReferenceConfigRecipeCoverage(t testingT, root map[string]interface{}) {
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, root, "entrypoints"),
		reflect.TypeOf(CanonicalEntrypoint{}),
		"entrypoints",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, root, "recipes"),
		reflect.TypeOf(CanonicalRecipe{}),
		"recipes",
	)
}

func assertReferenceConfigRoutingCoverage(t testingT, root map[string]interface{}) {
	routing := mustMapAt(t, root, "routing")

	// Recipe-wide policies can be demonstrated by a named recipe without
	// changing the broad default profile's compatibility behavior.
	profiles := []interface{}{routing}
	for _, profile := range collectChildMapsFromSlice(t, mustSliceAt(t, root, "recipes"), "routing", "recipes") {
		profiles = append(profiles, profile)
	}
	assertSliceUnionCoversStructFields(t, profiles, reflect.TypeOf(CanonicalRouting{}), "routing profiles")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, profiles, "fallback", "routing profiles"), reflect.TypeOf(fallback.FallbackPolicy{}), "routing.fallback")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, collectChildMapsFromSlice(t, profiles, "fallback", "routing profiles"), "circuit_breaker", "routing profiles"), reflect.TypeOf(fallback.CircuitBreakerConfig{}), "routing.fallback.circuit_breaker")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, profiles, "candidate_requirements", "routing profiles"), reflect.TypeOf(CandidateRequirements{}), "routing.candidate_requirements")
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
	assertReferenceConfigSignalCoverage(t, mustMapAt(t, routing, "signals"), collectChildMapsFromSlice(t, profiles, "signals", "routing profiles"))
	assertReferenceConfigProjectionCoverage(t, mustMapAt(t, routing, "projections"))
	assertReferenceConfigDecisionCoverage(t, referenceRecipeDecisions(t, root))
	recipeBindings := collectChildMapsFromSlice(t, profiles, "model_bindings", "routing profiles")
	for _, bindings := range recipeBindings {
		assertReferenceConfigBindingsBySignalType(t, bindings, "routing profiles[].model_bindings")
	}
}

func assertReferenceConfigSignalCoverage(t testingT, signals map[string]interface{}, profileSignals []map[string]interface{}) {
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
	// The rules the default routing profile reads score their hard and easy
	// candidates; a rule that names a deployment lives in a recipe whose
	// decisions do not read it.
	assertReferenceConfigComplexityCoverage(t, collectNestedSliceItems(t, profileSignals, "complexity", "routing profiles[].signals"))
	assertReferenceConfigModelRuleCoverage(t, signals)
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "modality"), reflect.TypeOf(ModalityRule{}), "routing.signals.modality")
	assertReferenceConfigRoleBindingCoverage(t, mustSliceAt(t, signals, "role_bindings"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "jailbreak"), reflect.TypeOf(JailbreakRule{}), "routing.signals.jailbreak")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "pii"), reflect.TypeOf(PIIRule{}), "routing.signals.pii")
	assertReferenceConfigKBSignalCoverage(t, mustSliceAt(t, signals, "kb"))
	assertReferenceConfigConversationSignalCoverage(t, mustSliceAt(t, signals, "conversation"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "events"), reflect.TypeOf(EventRule{}), "routing.signals.events")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "input_modality"), reflect.TypeOf(InputModalityRule{}), "routing.signals.input_modality")
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
	// hard_below/easy_above express a score that falls as difficulty rises,
	// which only a score.v1 backend can produce: the local margin is
	// hard-minus-easy, so a higher value is harder by construction. Covering
	// them here would mean shipping a reference rule whose direction is
	// wrong for the path the reference config actually uses.
	assertSliceUnionCoversStructFields(t, complexity, reflect.TypeOf(ComplexityRule{}), "routing.signals.complexity", "hard_below", "easy_above")
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

// assertReferenceConfigModelRuleCoverage covers the rules that ask their own
// question and name the deployment that answers it. A classifier rule's type,
// model, model_path and use_cpu, and a safety rule's or hazard part's model,
// are the deprecated way to choose that deployment.
func assertReferenceConfigModelRuleCoverage(t testingT, signals map[string]interface{}) {
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "classifiers"), reflect.TypeOf(ClassifierSignalRule{}), "routing.signals.classifiers", "type", "model", "model_path", "use_cpu")
	safety := mustSliceAt(t, signals, "safety")
	assertSliceUnionCoversStructFields(t, safety, reflect.TypeOf(SafetyRule{}), "routing.signals.safety", "model")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, safety, "hazard", "routing.signals.safety"),
		reflect.TypeOf(SafetyHazardRule{}),
		"routing.signals.safety[].hazard",
		"model",
	)
	systemOne := mustSliceAt(t, signals, "systemone")
	assertSliceUnionCoversStructFields(t, systemOne, reflect.TypeOf(SystemOneRule{}), "routing.signals.systemone")
	// No deployment reads audio yet, so the reference configuration cannot
	// show input.audio.
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, systemOne, "input", "routing.signals.systemone"),
		reflect.TypeOf(SystemOneInput{}),
		"routing.signals.systemone[].input",
		"audio",
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

func assertReferenceConfigConversationSignalCoverage(t testingT, conv []interface{}) {
	assertSliceUnionCoversStructFields(t, conv, reflect.TypeOf(ConversationRule{}), "routing.signals.conversation")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, conv, "feature", "routing.signals.conversation"),
		reflect.TypeOf(ConversationFeature{}),
		"routing.signals.conversation[].feature",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(
			t,
			collectChildMapsFromSlice(t, conv, "feature", "routing.signals.conversation"),
			"source", "routing.signals.conversation[].feature",
		),
		reflect.TypeOf(ConversationSource{}),
		"routing.signals.conversation[].feature.source",
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
