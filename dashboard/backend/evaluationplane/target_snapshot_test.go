package evaluationplane

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMixtureSnapshotGroupsAliasesAndScopesRecipePools(t *testing.T) {
	snapshot, err := ModelArmSnapshotFromYAML([]byte(multiRecipeMixtureTestYAML), "revision")
	if err != nil {
		t.Fatalf("ModelArmSnapshotFromYAML: %v", err)
	}
	if len(snapshot.Mixtures) != 2 {
		t.Fatalf("mixtures=%#v, want default and privacy recipes", snapshot.Mixtures)
	}
	byRecipe := make(map[string]MixtureTargetSnapshot, len(snapshot.Mixtures))
	for _, mixture := range snapshot.Mixtures {
		byRecipe[mixture.Mixture.RecipeName] = mixture
	}
	defaultMixture := byRecipe["default"].Mixture
	if defaultMixture.EntrypointModel != "primary-auto" ||
		!reflect.DeepEqual(defaultMixture.Aliases, []string{"primary-auto", "default-alias"}) ||
		len(defaultMixture.SupportModels) != 1 || defaultMixture.SupportModels[0].Model != "selector" ||
		!digestPattern.MatchString(defaultMixture.SupportModels[0].ProviderModelIDDigest) ||
		!digestPattern.MatchString(defaultMixture.SupportModels[0].ConfigDigest) ||
		defaultMixture.SupportModels[0].RuntimeRevision == nil ||
		*defaultMixture.SupportModels[0].RuntimeRevision != "revision" ||
		!digestPattern.MatchString(defaultMixture.SupportModels[0].BackendTopologyDigest) ||
		!digestPattern.MatchString(defaultMixture.SelectorPolicyDigest) ||
		!digestPattern.MatchString(defaultMixture.SelectorDigest) ||
		!digestPattern.MatchString(defaultMixture.AdaptationDigest) ||
		len(defaultMixture.ModelArms) != 2 || len(defaultMixture.Decisions) != 1 ||
		defaultMixture.Decisions[0].Algorithm != "prompt" {
		t.Fatalf("default mixture was not frozen from its aliases, pool, and selector: %#v", defaultMixture)
	}
	privacy := byRecipe["privacy"].Mixture
	if privacy.EntrypointModel != "privacy-primary" ||
		!reflect.DeepEqual(privacy.Aliases, []string{"privacy-primary", "privacy-alias"}) ||
		privacy.RecipeDescription != "Private lane" ||
		len(privacy.ModelArms) != 2 || privacy.FallbackArmID == "" || len(privacy.Decisions) != 1 ||
		len(privacy.Decisions[0].ArmIDs) != 1 || privacy.Decisions[0].Algorithm != "single" {
		t.Fatalf("named recipe aliases or scoped pool are invalid: %#v", privacy)
	}
	models := []string{privacy.ModelArms[0].Model, privacy.ModelArms[1].Model}
	if !reflect.DeepEqual(models, []string{"model-a", "model-c"}) {
		t.Fatalf("privacy pool leaked another recipe's candidate models: %v", models)
	}
	if defaultMixture.ID == privacy.ID || defaultMixture.RecipeDigest == privacy.RecipeDigest ||
		defaultMixture.PoolDigest == privacy.PoolDigest {
		t.Fatalf("recipe-scoped target identities were not isolated: default=%#v privacy=%#v", defaultMixture, privacy)
	}
}

func TestDecisionAlgorithmNameMatchesExtProcRuntimeSelectionProjection(t *testing.T) {
	models := []routerconfig.ModelRef{{Model: "a"}, {Model: "b"}}
	fastResponse := routerconfig.Decision{
		ModelRefs: models,
		Plugins: []routerconfig.DecisionPlugin{{
			Type: routerconfig.DecisionPluginFastResponse,
			Configuration: routerconfig.MustStructuredPayload(routerconfig.FastResponsePluginConfig{
				Message: "done",
			}),
		}},
	}
	tests := []struct {
		name     string
		decision routerconfig.Decision
		want     string
	}{
		{name: "provider default", decision: routerconfig.Decision{}, want: "default"},
		{name: "one candidate", decision: routerconfig.Decision{ModelRefs: models[:1]}, want: "single"},
		{name: "implicit static", decision: routerconfig.Decision{ModelRefs: models}, want: "static"},
		{name: "fast response", decision: fastResponse, want: "fast_response"},
		{name: "fast response with one candidate", decision: routerconfig.Decision{
			ModelRefs: models[:1], Plugins: fastResponse.Plugins,
		}, want: "fast_response"},
		{name: "fast response with provider default", decision: routerconfig.Decision{
			Plugins: fastResponse.Plugins,
		}, want: "fast_response"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, valid := decisionAlgorithmName(test.decision)
			if !valid || got != test.want {
				t.Fatalf("selection projection=%q, want ExtProc method %q", got, test.want)
			}
		})
	}
}

func TestDecisionAlgorithmNamePreservesCanonicalRouterCatalog(t *testing.T) {
	models := []routerconfig.ModelRef{{Model: "a"}, {Model: "b"}}
	for _, algorithm := range routerconfig.DecisionAlgorithmCatalog() {
		t.Run(algorithm.Type, func(t *testing.T) {
			got, valid := decisionAlgorithmName(routerconfig.Decision{
				ModelRefs: models,
				Algorithm: &routerconfig.AlgorithmConfig{Type: algorithm.Type},
			})
			if !valid || got != algorithm.Type {
				t.Fatalf("selection projection=(%q, %v), want exact Router algorithm %q", got, valid, algorithm.Type)
			}
			if !validMixtureDecisionAlgorithm(algorithm.Type) {
				t.Fatalf("target contract rejects canonical Router algorithm %q", algorithm.Type)
			}
		})
	}
}

func TestResolveMixtureDecisionsFailsClosedForUnknownAlgorithm(t *testing.T) {
	recipe := &routerconfig.RoutingRecipe{Profile: routerconfig.RoutingProfile{
		Decisions: []routerconfig.Decision{{
			Name:      "unknown",
			ModelRefs: []routerconfig.ModelRef{{Model: "a"}, {Model: "b"}},
			Algorithm: &routerconfig.AlgorithmConfig{Type: "unknown"},
		}},
	}}
	decisions, ready := resolveMixtureDecisions(
		recipe,
		[][]string{{"a", "b"}},
		map[string]string{"a": "arm-a", "b": "arm-b"},
	)
	if ready || len(decisions) != 0 {
		t.Fatalf("unknown algorithm produced ready snapshot decisions: ready=%v decisions=%#v", ready, decisions)
	}
	if validMixtureDecisionAlgorithm("unknown") {
		t.Fatal("target contract accepted an algorithm outside the canonical Router catalog")
	}
}

func TestDecisionWithoutModelRefsFreezesProviderDefaultArm(t *testing.T) {
	snapshot, err := ModelArmSnapshotFromYAML([]byte(providerDefaultDecisionTestYAML), "revision")
	if err != nil {
		t.Fatalf("ModelArmSnapshotFromYAML: %v", err)
	}
	mixture := requireSingleMixture(t, snapshot)
	if !mixture.Ready || mixture.Mixture.FallbackArmID == "" || len(mixture.Mixture.Decisions) != 1 {
		t.Fatalf("provider-default mixture is incomplete: %#v", mixture)
	}
	decision := mixture.Mixture.Decisions[0]
	if decision.Algorithm != "default" ||
		!reflect.DeepEqual(decision.ArmIDs, []string{mixture.Mixture.FallbackArmID}) {
		t.Fatalf("default decision boundary=%#v, fallback=%q", decision, mixture.Mixture.FallbackArmID)
	}
}

func TestModelArmSnapshotFromYAMLIsCanonicalDeterministicAndConnectivityFree(t *testing.T) {
	runtimeRevision := "git-test-revision"
	snapshot, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), runtimeRevision)
	if err != nil {
		t.Fatalf("ModelArmSnapshotFromYAML() error = %v", err)
	}
	if snapshot.ConfigDigest != digestBytes([]byte(modelArmTestYAML)) {
		t.Fatalf("config digest = %q, want digest of the parsed bytes", snapshot.ConfigDigest)
	}
	mixture := requireSingleMixture(t, snapshot)
	if !digestPattern.MatchString(mixture.Mixture.RecipeDigest) {
		t.Fatalf("recipe snapshot digest = %q, want a sha256 digest", mixture.Mixture.RecipeDigest)
	}
	if len(mixture.Mixture.ModelArms) != 2 {
		t.Fatalf("model arms = %#v, want only the two recipe-reachable, executable USD providers", mixture.Mixture.ModelArms)
	}

	fast := mixture.Mixture.ModelArms[0]
	if fast.Model != "Org/Fast Model" {
		t.Fatalf("first model = %q, want deterministic lexical order", fast.Model)
	}
	if fast.ProviderModelIDDigest != digestString("PrivateOrg/Secret-Upstream-ID") {
		t.Fatalf("provider digest = %q, want one-way identity digest", fast.ProviderModelIDDigest)
	}
	if fast.InputCostPerMillionTokensUSD != 1.25 || fast.OutputCostPerMillionTokensUSD != 4.5 {
		t.Fatalf("pricing = (%v, %v), want (1.25, 4.5)", fast.InputCostPerMillionTokensUSD, fast.OutputCostPerMillionTokensUSD)
	}
	if !reflect.DeepEqual(fast.Capabilities, []string{"chat", "vision"}) {
		t.Fatalf("capabilities = %#v, want sorted unique values", fast.Capabilities)
	}
	if !reflect.DeepEqual(fast.Modalities, []string{"text", "image"}) {
		t.Fatalf("modalities = %#v, want AR text plus declared vision", fast.Modalities)
	}
	if fast.ContextWindowTokens == nil || *fast.ContextWindowTokens != 32768 {
		t.Fatalf("context window = %#v, want 32768", fast.ContextWindowTokens)
	}
	if fast.ParameterSize == nil || *fast.ParameterSize != "7B" {
		t.Fatalf("parameter size = %#v, want 7B", fast.ParameterSize)
	}
	if fast.RuntimeRevision == nil || *fast.RuntimeRevision != runtimeRevision {
		t.Fatalf("runtime revision = %#v, want %q", fast.RuntimeRevision, runtimeRevision)
	}
	if fast.ConfigDigest == nil || !digestPattern.MatchString(*fast.ConfigDigest) || *fast.ConfigDigest == snapshot.ConfigDigest {
		t.Fatalf("arm config digest = %#v, want a provider-scoped digest distinct from full config %q", fast.ConfigDigest, snapshot.ConfigDigest)
	}
	if !digestPattern.MatchString(mixture.BackendTopologyDigest) {
		t.Fatalf("backend topology digest = %q", mixture.BackendTopologyDigest)
	}

	omni := mixture.Mixture.ModelArms[1]
	if omni.Model != "local/omni" {
		t.Fatalf("second model = %q, want local/omni", omni.Model)
	}
	if omni.ProviderModelIDDigest != digestString("local/omni") {
		t.Fatalf("fallback provider digest = %q, want digest of logical model", omni.ProviderModelIDDigest)
	}
	if !reflect.DeepEqual(omni.Modalities, []string{"text", "image", "document", "audio", "video"}) {
		t.Fatalf("omni modalities = %#v", omni.Modalities)
	}

	encoded, err := json.Marshal(mixture.Mixture)
	if err != nil {
		t.Fatalf("marshal arms: %v", err)
	}
	for _, privateValue := range []string{
		"PrivateOrg/Secret-Upstream-ID",
		"private.models.example.test",
		"local-private.example.test",
		"literal-test-secret",
	} {
		if strings.Contains(string(encoded), privateValue) {
			t.Fatalf("public arm snapshot leaked private value %q: %s", privateValue, encoded)
		}
	}
}

func TestLoadModelArmSnapshotUsesFileBytesAndSupportsAbsentConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(path, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write config fixture: %v", err)
	}

	snapshot, err := LoadModelArmSnapshot(path, "revision")
	if err != nil {
		t.Fatalf("LoadModelArmSnapshot() error = %v", err)
	}
	if snapshot.ConfigDigest != digestBytes([]byte(modelArmTestYAML)) || len(requireSingleMixture(t, snapshot).Mixture.ModelArms) != 2 {
		t.Fatalf("snapshot = %#v, want digest and arms from file bytes", snapshot)
	}

	absent, err := LoadModelArmSnapshot("  ", "revision")
	if err != nil {
		t.Fatalf("empty config path error = %v", err)
	}
	if absent.ConfigDigest != emptyConfigDigest || len(absent.Mixtures) != 0 {
		t.Fatalf("empty config snapshot = %#v", absent)
	}
}

func TestSnapshotSeparatesPolicyArmAndBackendTopologyOwnership(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision-a")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	baseMixture := requireSingleMixture(t, base)
	policyYAML := strings.Replace(modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
	policy, err := ModelArmSnapshotFromYAML([]byte(policyYAML), "revision-a")
	if err != nil {
		t.Fatalf("policy snapshot: %v", err)
	}
	policyMixture := requireSingleMixture(t, policy)
	if base.ConfigDigest == policy.ConfigDigest || baseMixture.Mixture.RecipeDigest == policyMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest != policyMixture.Mixture.BindingDigest ||
		baseMixture.BackendTopologyDigest != policyMixture.BackendTopologyDigest ||
		!sameModelArms(baseMixture.Mixture.ModelArms, policyMixture.Mixture.ModelArms) {
		t.Fatalf("policy-only change crossed ownership: base=%#v policy=%#v", base, policy)
	}

	endpointYAML := strings.Replace(modelArmTestYAML, "private.models.example.test:8000", "replacement.models.example.test:9000", 1)
	endpoint, err := ModelArmSnapshotFromYAML([]byte(endpointYAML), "revision-a")
	if err != nil {
		t.Fatalf("endpoint snapshot: %v", err)
	}
	endpointMixture := requireSingleMixture(t, endpoint)
	if baseMixture.BackendTopologyDigest == endpointMixture.BackendTopologyDigest ||
		baseMixture.Mixture.RecipeDigest != endpointMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest != endpointMixture.Mixture.BindingDigest ||
		!sameModelArms(baseMixture.Mixture.ModelArms, endpointMixture.Mixture.ModelArms) {
		t.Fatalf("endpoint-only change must affect only topology identity: base=%#v endpoint=%#v", base, endpoint)
	}

	priceYAML := strings.Replace(modelArmTestYAML, "prompt_per_1m: 1.25", "prompt_per_1m: 1.50", 1)
	price, err := ModelArmSnapshotFromYAML([]byte(priceYAML), "revision-a")
	if err != nil {
		t.Fatalf("price snapshot: %v", err)
	}
	priceMixture := requireSingleMixture(t, price)
	if baseMixture.Mixture.RecipeDigest != priceMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest != priceMixture.Mixture.BindingDigest ||
		baseMixture.BackendTopologyDigest != priceMixture.BackendTopologyDigest ||
		*baseMixture.Mixture.ModelArms[0].ConfigDigest == *priceMixture.Mixture.ModelArms[0].ConfigDigest ||
		baseMixture.Mixture.PoolDigest == priceMixture.Mixture.PoolDigest {
		t.Fatalf("arm price change did not alter only pool semantics: base=%#v price=%#v", base, price)
	}

	capabilityYAML := strings.Replace(modelArmTestYAML, "capabilities: [vision, chat, vision, \"\"]", "capabilities: [vision, chat, reasoning]", 1)
	capability, err := ModelArmSnapshotFromYAML([]byte(capabilityYAML), "revision-a")
	if err != nil {
		t.Fatalf("capability snapshot: %v", err)
	}
	capabilityMixture := requireSingleMixture(t, capability)
	if baseMixture.Mixture.RecipeDigest != capabilityMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest != capabilityMixture.Mixture.BindingDigest ||
		baseMixture.BackendTopologyDigest != capabilityMixture.BackendTopologyDigest ||
		*baseMixture.Mixture.ModelArms[0].ConfigDigest == *capabilityMixture.Mixture.ModelArms[0].ConfigDigest ||
		baseMixture.Mixture.PoolDigest == capabilityMixture.Mixture.PoolDigest {
		t.Fatalf("arm capability change did not alter only pool semantics: base=%#v capability=%#v", base, capability)
	}

	revision, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision-b")
	if err != nil {
		t.Fatalf("revision snapshot: %v", err)
	}
	revisionMixture := requireSingleMixture(t, revision)
	if baseMixture.Mixture.RecipeDigest != revisionMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest != revisionMixture.Mixture.BindingDigest ||
		*baseMixture.Mixture.ModelArms[0].ConfigDigest != *revisionMixture.Mixture.ModelArms[0].ConfigDigest ||
		baseMixture.Mixture.PoolDigest == revisionMixture.Mixture.PoolDigest {
		t.Fatalf("runtime revision must alter pool snapshot without changing arm config digest")
	}

	membershipYAML := strings.Replace(modelArmTestYAML, "\n        - model: local/omni", "", 1)
	membership, err := ModelArmSnapshotFromYAML([]byte(membershipYAML), "revision-a")
	if err != nil {
		t.Fatalf("membership snapshot: %v", err)
	}
	membershipMixture := requireSingleMixture(t, membership)
	if baseMixture.Mixture.RecipeDigest != membershipMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.BindingDigest == membershipMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.PoolDigest == membershipMixture.Mixture.PoolDigest {
		t.Fatalf("candidate membership must change binding and pool without changing policy: base=%#v candidate=%#v", baseMixture.Mixture, membershipMixture.Mixture)
	}
}

func TestSelectorSupportIdentityIsBindingOwnedAndEnvironmentOrthogonal(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(multiRecipeMixtureTestYAML), "selector-runtime-v1")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	baseMixture := mixtureForRecipe(t, base, "default")
	if len(baseMixture.Mixture.SupportModels) != 1 {
		t.Fatalf("base support models=%#v, want one frozen selector", baseMixture.Mixture.SupportModels)
	}

	providerYAML := strings.Replace(
		multiRecipeMixtureTestYAML,
		"    - name: selector\n      backend_refs:",
		"    - name: selector\n      provider_model_id: selector-runtime-v2\n      backend_refs:",
		1,
	)
	provider, err := ModelArmSnapshotFromYAML([]byte(providerYAML), "selector-runtime-v1")
	if err != nil {
		t.Fatalf("provider snapshot: %v", err)
	}
	providerMixture := mixtureForRecipe(t, provider, "default")
	if baseMixture.Mixture.BindingDigest != providerMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.RecipeDigest != providerMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.PoolDigest != providerMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != providerMixture.BackendTopologyDigest ||
		baseMixture.Mixture.SelectorPolicyDigest != providerMixture.Mixture.SelectorPolicyDigest ||
		baseMixture.Mixture.SelectorDigest == providerMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.AdaptationDigest != providerMixture.Mixture.AdaptationDigest ||
		baseMixture.Mixture.SupportModels[0].ProviderModelIDDigest == providerMixture.Mixture.SupportModels[0].ProviderModelIDDigest ||
		baseMixture.Mixture.SupportModels[0].ConfigDigest == providerMixture.Mixture.SupportModels[0].ConfigDigest {
		t.Fatalf("selector provider drift crossed factor ownership: base=%#v provider=%#v", baseMixture, providerMixture)
	}

	endpointYAML := strings.Replace(
		multiRecipeMixtureTestYAML,
		"selector.test:8000",
		"replacement-selector.test:9000",
		1,
	)
	endpoint, err := ModelArmSnapshotFromYAML([]byte(endpointYAML), "selector-runtime-v1")
	if err != nil {
		t.Fatalf("endpoint snapshot: %v", err)
	}
	endpointMixture := mixtureForRecipe(t, endpoint, "default")
	if baseMixture.Mixture.BindingDigest != endpointMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.RecipeDigest != endpointMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.PoolDigest != endpointMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != endpointMixture.BackendTopologyDigest ||
		baseMixture.Mixture.SelectorPolicyDigest != endpointMixture.Mixture.SelectorPolicyDigest ||
		baseMixture.Mixture.SelectorDigest == endpointMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.AdaptationDigest != endpointMixture.Mixture.AdaptationDigest ||
		baseMixture.Mixture.SupportModels[0].ConfigDigest != endpointMixture.Mixture.SupportModels[0].ConfigDigest ||
		baseMixture.Mixture.SupportModels[0].BackendTopologyDigest == endpointMixture.Mixture.SupportModels[0].BackendTopologyDigest {
		t.Fatalf("selector endpoint drift was not isolated to the selector: base=%#v endpoint=%#v", baseMixture, endpointMixture)
	}
}

func TestRecipeSelectorAndAdaptationDigestsAreOrthogonal(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(multiRecipeMixtureTestYAML), "revision")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	baseMixture := mixtureForRecipe(t, base, "default")

	selectorYAML := strings.Replace(
		multiRecipeMixtureTestYAML,
		"Choose one candidate.",
		"Choose the most capable candidate.",
		1,
	)
	selector, err := ModelArmSnapshotFromYAML([]byte(selectorYAML), "revision")
	if err != nil {
		t.Fatalf("selector snapshot: %v", err)
	}
	selectorMixture := mixtureForRecipe(t, selector, "default")
	if baseMixture.Mixture.SelectorPolicyDigest == selectorMixture.Mixture.SelectorPolicyDigest ||
		baseMixture.Mixture.SelectorDigest == selectorMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.RecipeDigest != selectorMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.AdaptationDigest != selectorMixture.Mixture.AdaptationDigest ||
		baseMixture.Mixture.BindingDigest != selectorMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.PoolDigest != selectorMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != selectorMixture.BackendTopologyDigest {
		t.Fatalf("selector policy crossed an orthogonal factor: base=%#v selector=%#v", baseMixture, selectorMixture)
	}

	adaptationYAML := strings.Replace(
		multiRecipeMixtureTestYAML,
		"      rules: {}\n      modelRefs:",
		"      rules: {}\n      adaptations: {mode: bypass}\n      modelRefs:",
		1,
	)
	if adaptationYAML == multiRecipeMixtureTestYAML {
		t.Fatal("adaptation fixture did not change")
	}
	adaptation, err := ModelArmSnapshotFromYAML([]byte(adaptationYAML), "revision")
	if err != nil {
		t.Fatalf("adaptation snapshot: %v", err)
	}
	adaptationMixture := mixtureForRecipe(t, adaptation, "default")
	if baseMixture.Mixture.AdaptationDigest == adaptationMixture.Mixture.AdaptationDigest ||
		baseMixture.Mixture.RecipeDigest != adaptationMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.SelectorDigest != adaptationMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.BindingDigest != adaptationMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.PoolDigest != adaptationMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != adaptationMixture.BackendTopologyDigest {
		t.Fatalf("adaptation crossed an orthogonal factor: base=%#v adaptation=%#v", baseMixture, adaptationMixture)
	}
}

func TestExternalClassifierExecutableIdentityBelongsToSelector(t *testing.T) {
	withExternal := strings.Replace(
		multiRecipeMixtureTestYAML,
		"  integrations:\n",
		"  model_catalog:\n"+
			"    external:\n"+
			"      - name: risk-judge\n"+
			"        llm_provider: openai\n"+
			"        model_role: classification\n"+
			"        llm_endpoint:\n"+
			"          address: classifier.private.test\n"+
			"          port: 443\n"+
			"          protocol: https\n"+
			"          use_chat_template: true\n"+
			"          prompt_template: '{{ .Prompt }}'\n"+
			"        llm_model_name: private/risk-model-v1\n"+
			"        llm_timeout_seconds: 5\n"+
			"        parser_type: json\n"+
			"        max_tokens: 32\n"+
			"  integrations:\n",
		1,
	)
	withExternal = strings.Replace(
		withExternal,
		"routing:\n  modelCards:",
		"routing:\n  signals:\n"+
			"    classifiers:\n"+
			"      - name: tool-risk\n"+
			"        type: llm\n"+
			"        model: risk-judge\n"+
			"        labels: [SAFE, RISKY]\n"+
			"        instructions: Classify the request.\n"+
			"  modelCards:",
		1,
	)
	base, err := ModelArmSnapshotFromYAML([]byte(withExternal), "revision")
	if err != nil {
		t.Fatalf("external classifier snapshot: %v", err)
	}
	baseMixture := mixtureForRecipe(t, base, "default")
	if len(baseMixture.Mixture.SupportModels) != 2 ||
		baseMixture.Mixture.SupportModels[0].Model != "risk-judge" ||
		baseMixture.Mixture.SupportModels[1].Model != "selector" {
		t.Fatalf("selector support identities=%#v, want classifier and prompt selector", baseMixture.Mixture.SupportModels)
	}
	encoded, err := json.Marshal(baseMixture.Mixture.SupportModels[0])
	if err != nil || strings.Contains(string(encoded), "private/risk-model-v1") ||
		strings.Contains(string(encoded), "classifier.private.test") {
		t.Fatalf("external classifier identity leaked private execution details: %s err=%v", encoded, err)
	}

	providerYAML := strings.Replace(withExternal, "private/risk-model-v1", "private/risk-model-v2", 1)
	provider, err := ModelArmSnapshotFromYAML([]byte(providerYAML), "revision")
	if err != nil {
		t.Fatalf("external classifier provider snapshot: %v", err)
	}
	providerMixture := mixtureForRecipe(t, provider, "default")
	if baseMixture.Mixture.SelectorDigest == providerMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.SelectorPolicyDigest != providerMixture.Mixture.SelectorPolicyDigest ||
		baseMixture.Mixture.BindingDigest != providerMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.RecipeDigest != providerMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.PoolDigest != providerMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != providerMixture.BackendTopologyDigest {
		t.Fatalf("classifier provider identity crossed factor ownership: base=%#v provider=%#v", baseMixture, providerMixture)
	}

	endpointYAML := strings.Replace(withExternal, "classifier.private.test", "replacement.private.test", 1)
	endpoint, err := ModelArmSnapshotFromYAML([]byte(endpointYAML), "revision")
	if err != nil {
		t.Fatalf("external classifier endpoint snapshot: %v", err)
	}
	endpointMixture := mixtureForRecipe(t, endpoint, "default")
	if baseMixture.Mixture.SelectorDigest == endpointMixture.Mixture.SelectorDigest ||
		baseMixture.Mixture.BindingDigest != endpointMixture.Mixture.BindingDigest ||
		baseMixture.Mixture.RecipeDigest != endpointMixture.Mixture.RecipeDigest ||
		baseMixture.Mixture.PoolDigest != endpointMixture.Mixture.PoolDigest ||
		baseMixture.BackendTopologyDigest != endpointMixture.BackendTopologyDigest {
		t.Fatalf("classifier endpoint identity crossed factor ownership: base=%#v endpoint=%#v", baseMixture, endpointMixture)
	}
}

func TestRecipeDescriptionIsPresentationMetadataNotPolicyTreatment(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(multiRecipeMixtureTestYAML), "revision")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	describedYAML := strings.Replace(multiRecipeMixtureTestYAML, "Private lane", "Documentation-only rename", 1)
	described, err := ModelArmSnapshotFromYAML([]byte(describedYAML), "revision")
	if err != nil {
		t.Fatalf("description snapshot: %v", err)
	}
	basePrivacy := mixtureForRecipe(t, base, "privacy")
	describedPrivacy := mixtureForRecipe(t, described, "privacy")
	if base.ConfigDigest == described.ConfigDigest ||
		basePrivacy.Mixture.RecipeDescription == describedPrivacy.Mixture.RecipeDescription ||
		basePrivacy.Mixture.RecipeDigest != describedPrivacy.Mixture.RecipeDigest ||
		basePrivacy.Mixture.BindingDigest != describedPrivacy.Mixture.BindingDigest ||
		basePrivacy.Mixture.PoolDigest != describedPrivacy.Mixture.PoolDigest ||
		basePrivacy.BackendTopologyDigest != describedPrivacy.BackendTopologyDigest {
		t.Fatalf("description metadata crossed an executable factor: base=%#v described=%#v", basePrivacy, describedPrivacy)
	}
}

func TestDecisionPresentationMetadataIsNotPolicyTreatment(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	metadataYAML := strings.Replace(
		modelArmTestYAML,
		"    - name: route\n      rules: {}",
		"    - name: route\n      description: Operator-facing explanation\n      annotations:\n        owner: evaluation\n      rules: {}",
		1,
	)
	metadata, err := ModelArmSnapshotFromYAML([]byte(metadataYAML), "revision")
	if err != nil {
		t.Fatalf("metadata snapshot: %v", err)
	}
	left := requireSingleMixture(t, base)
	right := requireSingleMixture(t, metadata)
	if base.ConfigDigest == metadata.ConfigDigest ||
		left.Mixture.RecipeDigest != right.Mixture.RecipeDigest ||
		left.Mixture.BindingDigest != right.Mixture.BindingDigest ||
		left.Mixture.PoolDigest != right.Mixture.PoolDigest ||
		left.BackendTopologyDigest != right.BackendTopologyDigest {
		t.Fatalf("decision presentation metadata crossed an executable factor: base=%#v metadata=%#v", left, right)
	}
}

func TestDecisionNameBelongsOnlyToRecipeIdentity(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	renamedYAML := strings.Replace(modelArmTestYAML, "    - name: route", "    - name: renamed-route", 1)
	renamed, err := ModelArmSnapshotFromYAML([]byte(renamedYAML), "revision")
	if err != nil {
		t.Fatalf("renamed snapshot: %v", err)
	}
	left := requireSingleMixture(t, base)
	right := requireSingleMixture(t, renamed)
	if left.Mixture.RecipeDigest == right.Mixture.RecipeDigest ||
		left.Mixture.SelectorDigest != right.Mixture.SelectorDigest ||
		left.Mixture.AdaptationDigest != right.Mixture.AdaptationDigest ||
		left.Mixture.BindingDigest != right.Mixture.BindingDigest ||
		left.Mixture.PoolDigest != right.Mixture.PoolDigest ||
		left.BackendTopologyDigest != right.BackendTopologyDigest {
		t.Fatalf("decision name crossed factor ownership: base=%#v renamed=%#v", left, right)
	}
}

func TestUnknownModelRuntimeRevisionDoesNotMutatePool(t *testing.T) {
	unknownA, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "")
	if err != nil {
		t.Fatalf("first snapshot: %v", err)
	}
	unknownB, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "")
	if err != nil {
		t.Fatalf("second snapshot: %v", err)
	}
	left := requireSingleMixture(t, unknownA).Mixture
	right := requireSingleMixture(t, unknownB).Mixture
	if left.PoolDigest != right.PoolDigest {
		t.Fatalf("unknown runtime revision changed pool identity: %q != %q", left.PoolDigest, right.PoolDigest)
	}
	for _, arm := range left.ModelArms {
		if arm.RuntimeRevision != nil {
			t.Fatalf("arm %q falsely claims runtime revision %q", arm.ID, *arm.RuntimeRevision)
		}
	}
}

func TestPolicySnapshotDigestUsesCanonicalPolicySubset(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	commentOnly := "# deployment comment\n" + modelArmTestYAML
	commented, err := ModelArmSnapshotFromYAML([]byte(commentOnly), "revision")
	if err != nil {
		t.Fatalf("commented snapshot: %v", err)
	}
	if base.ConfigDigest == commented.ConfigDigest ||
		requireSingleMixture(t, base).Mixture.RecipeDigest != requireSingleMixture(t, commented).Mixture.RecipeDigest {
		t.Fatalf("raw lineage and canonical policy identities were not separated: base=%#v commented=%#v", base, commented)
	}
}

func TestPortableModelArmIDIsStablePortableAndCollisionResistant(t *testing.T) {
	portable := regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	models := []string{
		"Org/Model With Spaces",
		"Org/Model@With@Spaces",
		strings.Repeat("Very-Long-Model/", 20),
		"模型/视觉",
	}
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		id := portableModelArmID(model)
		if id != portableModelArmID(model) {
			t.Fatalf("ID for %q is not stable", model)
		}
		if !portable.MatchString(id) {
			t.Fatalf("ID %q for %q is not portable", id, model)
		}
		if _, duplicate := seen[id]; duplicate {
			t.Fatalf("distinct model %q collided at ID %q", model, id)
		}
		seen[id] = struct{}{}
	}
}

func TestValidProviderArmRequiresBackendAndTruthfulUSDPricing(t *testing.T) {
	backend := []routerconfig.CanonicalBackendRef{{Endpoint: "private.test:8000"}}
	tests := []struct {
		name     string
		provider routerconfig.CanonicalProviderModel
		want     bool
	}{
		{
			name: "explicit USD",
			provider: routerconfig.CanonicalProviderModel{
				Name: "paid", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{Currency: "Usd", PromptPer1M: 1, CompletionPer1M: 2},
			},
			want: true,
		},
		{
			name:     "zero-cost local arm",
			provider: routerconfig.CanonicalProviderModel{Name: "local", BackendRefs: backend},
			want:     true,
		},
		{
			name: "metadata only",
			provider: routerconfig.CanonicalProviderModel{
				Name: "metadata", Pricing: routerconfig.ModelPricing{Currency: "USD"},
			},
		},
		{
			name: "omitted currency defaults to USD",
			provider: routerconfig.CanonicalProviderModel{
				Name: "ambiguous", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{PromptPer1M: 1},
			},
			want: true,
		},
		{
			name: "non USD",
			provider: routerconfig.CanonicalProviderModel{
				Name: "foreign", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{Currency: "EUR"},
			},
		},
		{
			name: "negative price",
			provider: routerconfig.CanonicalProviderModel{
				Name: "negative", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{Currency: "USD", PromptPer1M: -1},
			},
		},
		{
			name: "non finite price",
			provider: routerconfig.CanonicalProviderModel{
				Name: "infinite", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{Currency: "USD", CompletionPer1M: math.Inf(1)},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := strings.TrimSpace(test.provider.Name)
			pricing := test.provider.Pricing
			if pricing.Currency == "" {
				pricing.Currency = "USD"
			}
			binding := mixtureModelBinding{EffectiveModel: model, BaseModel: model}
			if got := validProviderArm(test.provider, binding, pricing); got != test.want {
				t.Fatalf("validProviderArm() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestModelArmSnapshotRejectsInvalidCanonicalConfig(t *testing.T) {
	if _, err := ModelArmSnapshotFromYAML([]byte("version: [invalid"), "revision"); err == nil {
		t.Fatal("invalid canonical YAML unexpectedly succeeded")
	}
}

func requireSingleMixture(t *testing.T, snapshot ModelArmSnapshot) MixtureTargetSnapshot {
	t.Helper()
	if len(snapshot.Mixtures) != 1 {
		t.Fatalf("mixture snapshots = %#v, want exactly one default recipe target", snapshot.Mixtures)
	}
	return snapshot.Mixtures[0]
}

func mixtureForRecipe(t *testing.T, snapshot ModelArmSnapshot, recipe string) MixtureTargetSnapshot {
	t.Helper()
	for _, mixture := range snapshot.Mixtures {
		if mixture.Mixture.RecipeName == recipe {
			return mixture
		}
	}
	t.Fatalf("recipe %q is unavailable in %#v", recipe, snapshot.Mixtures)
	return MixtureTargetSnapshot{}
}
