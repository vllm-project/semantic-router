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
	assertReferenceConfigSignalCoverage(t, mustMapAt(t, routing, "signals"))
	assertReferenceConfigProjectionCoverage(t, mustMapAt(t, routing, "projections"))
	assertReferenceConfigDecisionCoverage(t, referenceRecipeDecisions(t, root))
	recipeBindings := collectChildMapsFromSlice(t, profiles, "model_bindings", "routing profiles")
	for _, bindings := range recipeBindings {
		assertReferenceConfigBindingsBySignalType(t, bindings, "routing profiles[].model_bindings")
	}
}

func assertReferenceConfigSignalCoverage(t testingT, signals map[string]interface{}) {
	assertMapCoversStructFields(t, signals, reflect.TypeOf(CanonicalSignals{}), "routing.signals")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "keywords"), reflect.TypeOf(KeywordRule{}), "routing.signals.keywords")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, signals, "domains"), "model_scores", "routing.signals.domains"),
		reflect.TypeOf(ModelScore{}),
		"routing.signals.domains[].model_scores",
	)
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "language"), reflect.TypeOf(LanguageRule{}), "routing.signals.language")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "context"), reflect.TypeOf(ContextRule{}), "routing.signals.context")
	assertReferenceConfigStructureCoverage(t, mustSliceAt(t, signals, "structure"))
	assertReferenceConfigApplicationRuleCoverage(t, signals)
	assertReferenceConfigModelRuleCoverage(t, signals)
	assertReferenceConfigRoleBindingCoverage(t, mustSliceAt(t, signals, "role_bindings"))
	assertReferenceConfigConversationSignalCoverage(t, mustSliceAt(t, signals, "conversation"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "events"), reflect.TypeOf(EventRule{}), "routing.signals.events")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "input_modality"), reflect.TypeOf(InputModalityRule{}), "routing.signals.input_modality")
}

// assertReferenceConfigApplicationRuleCoverage covers every field of the
// application signals, which ask the Router's presets, with the inputs the
// rules may add.
func assertReferenceConfigApplicationRuleCoverage(t testingT, signals map[string]interface{}) {
	for key, rule := range map[string]reflect.Type{
		"domains":        reflect.TypeOf(Category{}),
		"fact_check":     reflect.TypeOf(FactCheckRule{}),
		"user_feedbacks": reflect.TypeOf(UserFeedbackRule{}),
		"jailbreak":      reflect.TypeOf(JailbreakRule{}),
		"safety":         reflect.TypeOf(SafetyRule{}),
		"pii":            reflect.TypeOf(PIIRule{}),
		"hallucination":  reflect.TypeOf(HallucinationRule{}),
		"refusal":        reflect.TypeOf(RefusalRule{}),
		"turn_relation":  reflect.TypeOf(TurnRelationRule{}),
		"complexity":     reflect.TypeOf(ComplexityRule{}),
		"modality":       reflect.TypeOf(ModalityRule{}),
	} {
		assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, key), rule, "routing.signals."+key)
	}
	for key, input := range map[string]reflect.Type{
		"jailbreak":     reflect.TypeOf(JailbreakRuleInput{}),
		"pii":           reflect.TypeOf(PIIRuleInput{}),
		"turn_relation": reflect.TypeOf(TurnRelationRuleInput{}),
	} {
		assertSliceUnionCoversStructFields(
			t,
			collectChildMapsFromSlice(t, mustSliceAt(t, signals, key), "input", "routing.signals."+key),
			input,
			"routing.signals."+key+"[].input",
		)
	}
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

// assertReferenceConfigModelRuleCoverage covers the systemone rules, which
// ask their own question of the deployment they name.
func assertReferenceConfigModelRuleCoverage(t testingT, signals map[string]interface{}) {
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
