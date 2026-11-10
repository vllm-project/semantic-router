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
	// The reference config writes no key inline (api_key).
	assertSliceUnionCoversStructFields(
		t,
		backendRefs,
		reflect.TypeOf(CanonicalBackendRef{}),
		"providers.models[].backend_refs",
		"api_key",
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
	// The reference config cannot show client_auth while normalization still
	// rejects it; it joins the reference once the Router enforces it.
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, root, "entrypoints"),
		reflect.TypeOf(CanonicalEntrypoint{}),
		"entrypoints",
		"client_auth",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, root, "recipes"),
		reflect.TypeOf(CanonicalRecipe{}),
		"recipes",
	)
}

func assertReferenceConfigRoutingCoverage(t testingT, root map[string]interface{}) {
	// Recipe-wide policies can be demonstrated by any recipe without changing
	// the broad default recipe's compatibility behavior.
	profiles := mustSliceAt(t, root, "recipes")
	defaultRecipe := referenceDefaultRecipe(t, profiles)
	models := mustSliceAt(t, root, "providers", "models")
	assertSliceUnionCoversStructFields(t, profiles, reflect.TypeOf(CanonicalRecipe{}), "recipes")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, profiles, "fallback", "recipes"), reflect.TypeOf(fallback.FallbackPolicy{}), "recipes[].fallback")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, collectChildMapsFromSlice(t, profiles, "fallback", "recipes"), "circuit_breaker", "recipes"), reflect.TypeOf(fallback.CircuitBreakerConfig{}), "recipes[].fallback.circuit_breaker")
	assertSliceUnionCoversStructFields(t, collectChildMapsFromSlice(t, profiles, "candidate_requirements", "recipes"), reflect.TypeOf(CandidateRequirements{}), "recipes[].candidate_requirements")
	assertSliceUnionCoversStructFields(
		t,
		models,
		reflect.TypeOf(ModelFacts{}),
		"providers.models",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, models, "loras", "providers.models"),
		reflect.TypeOf(LoRAAdapter{}),
		"providers.models[].loras",
	)
	assertReferenceConfigSignalCoverage(t, mustMapAt(t, defaultRecipe, "signals"))
	assertReferenceConfigProjectionCoverage(t, mustMapAt(t, defaultRecipe, "projections"))
	assertReferenceConfigDecisionCoverage(t, referenceRecipeDecisions(t, root))
	recipeBindings := collectChildMapsFromSlice(t, profiles, "model_bindings", "recipes")
	for _, bindings := range recipeBindings {
		assertReferenceConfigBindingsBySignalType(t, bindings, "recipes[].model_bindings")
	}
}

func assertReferenceConfigSignalCoverage(t testingT, signals map[string]interface{}) {
	assertMapCoversStructFields(t, signals, reflect.TypeOf(CanonicalSignals{}), "recipes[default].signals")
	keywords := mustSliceAt(t, signals, "keywords")
	assertSliceUnionCoversStructFields(t, keywords, reflect.TypeOf(KeywordRule{}), "recipes[default].signals.keywords")
	assertSliceUnionCoversStructFields(
		t,
		append(
			collectChildMapsFromSlice(t, keywords, "count", "recipes[default].signals.keywords"),
			collectChildMapsFromSlice(t, keywords, "density", "recipes[default].signals.keywords")...,
		),
		reflect.TypeOf(NumericPredicate{}),
		"recipes[default].signals.keywords[].count and density",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, signals, "domains"), "model_scores", "recipes[default].signals.domains"),
		reflect.TypeOf(ModelScore{}),
		"recipes[default].signals.domains[].model_scores",
	)
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "language"), reflect.TypeOf(LanguageRule{}), "recipes[default].signals.language")
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "context"), reflect.TypeOf(ContextRule{}), "recipes[default].signals.context")
	assertReferenceConfigApplicationRuleCoverage(t, signals)
	assertReferenceConfigModelRuleCoverage(t, signals)
	assertReferenceConfigRoleBindingCoverage(t, mustSliceAt(t, signals, "role_bindings"))
	assertReferenceConfigConversationSignalCoverage(t, mustSliceAt(t, signals, "conversation"))
	assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, "input_modality"), reflect.TypeOf(InputModalityRule{}), "recipes[default].signals.input_modality")
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
		assertSliceUnionCoversStructFields(t, mustSliceAt(t, signals, key), rule, "recipes[default].signals."+key)
	}
	for key, input := range map[string]reflect.Type{
		"jailbreak":     reflect.TypeOf(JailbreakRuleInput{}),
		"pii":           reflect.TypeOf(PIIRuleInput{}),
		"turn_relation": reflect.TypeOf(TurnRelationRuleInput{}),
	} {
		assertSliceUnionCoversStructFields(
			t,
			collectChildMapsFromSlice(t, mustSliceAt(t, signals, key), "input", "recipes[default].signals."+key),
			input,
			"recipes[default].signals."+key+"[].input",
		)
	}
}

func assertReferenceConfigProjectionCoverage(t testingT, projections map[string]interface{}) {
	assertMapCoversStructFields(t, projections, reflect.TypeOf(CanonicalProjections{}), "recipes[default].projections")
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "partitions"),
		reflect.TypeOf(ProjectionPartition{}),
		"recipes[default].projections.partitions",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "scores"),
		reflect.TypeOf(ProjectionScore{}),
		"recipes[default].projections.scores",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, projections, "scores"), "inputs", "recipes[default].projections.scores"),
		reflect.TypeOf(ProjectionScoreInput{}),
		"recipes[default].projections.scores[].inputs",
	)
	assertSliceUnionCoversStructFields(
		t,
		mustSliceAt(t, projections, "mappings"),
		reflect.TypeOf(ProjectionMapping{}),
		"recipes[default].projections.mappings",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, mustSliceAt(t, projections, "mappings"), "calibration", "recipes[default].projections.mappings"),
		reflect.TypeOf(ProjectionMappingCalibration{}),
		"recipes[default].projections.mappings[].calibration",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, mustSliceAt(t, projections, "mappings"), "outputs", "recipes[default].projections.mappings"),
		reflect.TypeOf(ProjectionMappingOutput{}),
		"recipes[default].projections.mappings[].outputs",
	)
}

// assertReferenceConfigModelRuleCoverage covers the systemone rules, which
// ask their own question of the deployment they name.
func assertReferenceConfigModelRuleCoverage(t testingT, signals map[string]interface{}) {
	systemOne := mustSliceAt(t, signals, "systemone")
	assertSliceUnionCoversStructFields(t, systemOne, reflect.TypeOf(SystemOneRule{}), "recipes[default].signals.systemone")
	// No deployment reads audio yet, so the reference configuration cannot
	// show input.audio.
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, systemOne, "input", "recipes[default].signals.systemone"),
		reflect.TypeOf(SystemOneInput{}),
		"recipes[default].signals.systemone[].input",
		"audio",
	)
}

func assertReferenceConfigRoleBindingCoverage(t testingT, roleBindings []interface{}) {
	assertSliceUnionCoversStructFields(t, roleBindings, reflect.TypeOf(RoleBinding{}), "recipes[default].signals.role_bindings")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, roleBindings, "subjects", "recipes[default].signals.role_bindings"),
		reflect.TypeOf(Subject{}),
		"recipes[default].signals.role_bindings[].subjects",
	)
}

func assertReferenceConfigConversationSignalCoverage(t testingT, conv []interface{}) {
	assertSliceUnionCoversStructFields(t, conv, reflect.TypeOf(ConversationRule{}), "recipes[default].signals.conversation")
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, conv, "feature", "recipes[default].signals.conversation"),
		reflect.TypeOf(ConversationFeature{}),
		"recipes[default].signals.conversation[].feature",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(
			t,
			collectChildMapsFromSlice(t, conv, "feature", "recipes[default].signals.conversation"),
			"source", "recipes[default].signals.conversation[].feature",
		),
		reflect.TypeOf(ConversationSource{}),
		"recipes[default].signals.conversation[].feature.source",
	)
}

func assertReferenceConfigDecisionCoverage(t testingT, decisions []interface{}) {
	assertSliceUnionCoversStructFields(t, decisions, reflect.TypeOf(Decision{}), "recipes[default].decisions")
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, decisions, "modelRefs", "recipes[default].decisions"),
		reflect.TypeOf(ModelRef{}),
		"recipes[default].decisions[].modelRefs",
	)
	candidateIterations := collectNestedSliceItems(t, decisions, "candidateIterations", "recipes[default].decisions")
	assertSliceUnionCoversStructFields(
		t,
		candidateIterations,
		reflect.TypeOf(CandidateIterationConfig{}),
		"recipes[default].decisions[].candidateIterations",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, candidateIterations, "models", "recipes[default].decisions[].candidateIterations"),
		reflect.TypeOf(ModelRef{}),
		"recipes[default].decisions[].candidateIterations[].models",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, candidateIterations, "outputs", "recipes[default].decisions[].candidateIterations"),
		reflect.TypeOf(CandidateIterationOutputConfig{}),
		"recipes[default].decisions[].candidateIterations[].outputs",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectChildMapsFromSlice(t, decisions, "algorithm", "recipes[default].decisions"),
		reflect.TypeOf(AlgorithmConfig{}),
		"recipes[default].decisions[].algorithm",
	)
	assertSliceUnionCoversStructFields(
		t,
		collectNestedSliceItems(t, decisions, "plugins", "recipes[default].decisions"),
		reflect.TypeOf(DecisionPlugin{}),
		"recipes[default].decisions[].plugins",
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
	assertReferenceConfigSignalsCompressionCoverage(t, mustSliceAt(t, global, "plugins"))
}
