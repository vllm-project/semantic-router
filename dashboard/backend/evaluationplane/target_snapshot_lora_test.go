package evaluationplane

import (
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const loraSnapshotTestYAML = `version: v0.3
global:
  router:
    auto_model_names: [lora-mom]
providers:
  defaults:
    default_model: general-expert
  models:
    - name: foundation-private
      provider_model_id: PrivateCorp/Foundation-Runtime
      backend_refs:
        - name: shared-private
          endpoint: foundation.private.test:8000/v1
          api_key: base-secret
      pricing:
        prompt_per_1m: 0.25
        completion_per_1m: 0.75
        cached_input_per_1m: 0.05
        cache_write_per_1m: 0.10
    - name: premium-expert
      pricing:
        prompt_per_1m: 1.5
        completion_per_1m: 3.5
routing:
  modelCards:
    - name: foundation-private
      param_size: 14B
      context_window_size: 65536
      capabilities: [chat, code]
      modality: ar
      loras:
        - {name: general-expert, description: General adapter}
        - {name: premium-expert, description: Premium adapter}
    - name: premium-expert
      capabilities: [vision]
      modality: image
  decisions:
    - name: experts
      rules: {}
      modelRefs:
        - {model: foundation-private, lora_name: general-expert}
        - {model: foundation-private, lora_name: premium-expert}
`

func TestLoRAArmsFreezeEffectiveIdentityAndBaseExecutionMetadata(t *testing.T) {
	mixture := mustLoRAMixture(t, loraSnapshotTestYAML)
	if len(mixture.Mixture.ModelArms) != 2 {
		t.Fatalf("LoRA pool=%#v, want two effective adapter arms", mixture.Mixture.ModelArms)
	}
	general := requireSnapshotArm(t, mixture, "general-expert")
	premium := requireSnapshotArm(t, mixture, "premium-expert")
	if general.ID == premium.ID || general.ProviderModelIDDigest == premium.ProviderModelIDDigest {
		t.Fatalf("distinct adapters collapsed into one identity: general=%#v premium=%#v", general, premium)
	}
	baseDigest := digestString("PrivateCorp/Foundation-Runtime")
	for _, arm := range []ModelArm{general, premium} {
		wantProviderDigest := digestJSON(adapterProviderIdentityFingerprint{
			BaseProviderModelIDDigest: baseDigest,
			AdapterIdentity:           arm.Model,
		})
		if arm.ProviderModelIDDigest != wantProviderDigest ||
			!reflect.DeepEqual(arm.Capabilities, []string{"chat", "code"}) ||
			!reflect.DeepEqual(arm.Modalities, []string{"text"}) ||
			arm.ContextWindowTokens == nil || *arm.ContextWindowTokens != 65536 ||
			arm.ParameterSize == nil || *arm.ParameterSize != "14B" {
			t.Fatalf("adapter arm did not inherit base provider/card identity: %#v", arm)
		}
	}
	if general.InputCostPerMillionTokensUSD != 0.25 || general.OutputCostPerMillionTokensUSD != 0.75 ||
		premium.InputCostPerMillionTokensUSD != 1.5 || premium.OutputCostPerMillionTokensUSD != 3.5 {
		t.Fatalf("LoRA inherited/override pricing is wrong: general=%#v premium=%#v", general, premium)
	}
	if mixture.Mixture.FallbackArmID != general.ID ||
		!reflect.DeepEqual(mixture.Mixture.Decisions[0].ArmIDs, sortedStrings(general.ID, premium.ID)) {
		t.Fatalf("fallback or decision did not bind effective adapters: %#v", mixture.Mixture)
	}
	for _, arm := range mixture.Mixture.ModelArms {
		if arm.Model == "foundation-private" {
			t.Fatalf("non-routable base leaked into the effective pool: %#v", mixture.Mixture.ModelArms)
		}
	}
	if err := validateMixtureContract(&mixture.Mixture); err != nil {
		t.Fatalf("LoRA mixture contract: %v", err)
	}
}

func TestLoRABaseAndAdaptersStayDistinctWhileSharingOneTopology(t *testing.T) {
	withBase := strings.Replace(
		loraSnapshotTestYAML,
		"      modelRefs:\n",
		"      modelRefs:\n        - {model: foundation-private}\n",
		1,
	)
	mixture := mustLoRAMixture(t, withBase)
	if len(mixture.Mixture.ModelArms) != 3 {
		t.Fatalf("base and adapters did not remain distinct: %#v", mixture.Mixture.ModelArms)
	}
	base := requireSnapshotArm(t, mixture, "foundation-private")
	general := requireSnapshotArm(t, mixture, "general-expert")
	if base.ID == general.ID || base.ProviderModelIDDigest == general.ProviderModelIDDigest {
		t.Fatalf("base and adapter identities collapsed: base=%#v adapter=%#v", base, general)
	}
	if len(mixture.Mixture.Decisions) != 1 || len(mixture.Mixture.Decisions[0].ArmIDs) != 3 {
		t.Fatalf("decision boundary lost an effective arm: %#v", mixture.Mixture.Decisions)
	}
	assertLoRATopologyUsesOnlyBase(t, withBase, mixture)
}

func TestLoRADirectAliasAndDefaultOnlyDecisionResolveBase(t *testing.T) {
	directAlias := strings.Replace(
		loraSnapshotTestYAML,
		"        - {model: foundation-private, lora_name: general-expert}\n"+
			"        - {model: foundation-private, lora_name: premium-expert}\n",
		"        - {model: general-expert}\n",
		1,
	)
	aliasMixture := mustLoRAMixture(t, directAlias)
	general := requireSnapshotArm(t, aliasMixture, "general-expert")
	if len(aliasMixture.Mixture.ModelArms) != 1 ||
		!reflect.DeepEqual(aliasMixture.Mixture.Decisions[0].ArmIDs, []string{general.ID}) {
		t.Fatalf("direct LoRA alias did not resolve its base: %#v", aliasMixture.Mixture)
	}

	defaultOnly := strings.Replace(
		loraSnapshotTestYAML,
		"      modelRefs:\n"+
			"        - {model: foundation-private, lora_name: general-expert}\n"+
			"        - {model: foundation-private, lora_name: premium-expert}\n",
		"",
		1,
	)
	defaultMixture := mustLoRAMixture(t, defaultOnly)
	defaultArm := requireSnapshotArm(t, defaultMixture, "general-expert")
	if len(defaultMixture.Mixture.ModelArms) != 1 ||
		defaultMixture.Mixture.FallbackArmID != defaultArm.ID ||
		!reflect.DeepEqual(defaultMixture.Mixture.Decisions[0].ArmIDs, []string{defaultArm.ID}) {
		t.Fatalf("LoRA default-only decision did not bind the effective alias: %#v", defaultMixture.Mixture)
	}
}

func TestLoRACandidateIterationUsesEffectiveIdentity(t *testing.T) {
	iterationYAML := strings.Replace(
		loraSnapshotTestYAML,
		"        - {model: foundation-private, lora_name: general-expert}\n"+
			"        - {model: foundation-private, lora_name: premium-expert}\n",
		"        - {model: foundation-private, lora_name: general-expert}\n"+
			"      candidateIterations:\n"+
			"        - variable: candidate\n"+
			"          source: models\n"+
			"          models:\n"+
			"            - {model: foundation-private, lora_name: premium-expert}\n"+
			"          outputs:\n"+
			"            - {type: model, value: candidate}\n"+
			"      algorithm: {type: static}\n",
		1,
	)
	mixture := mustLoRAMixture(t, iterationYAML)
	general := requireSnapshotArm(t, mixture, "general-expert")
	premium := requireSnapshotArm(t, mixture, "premium-expert")
	if len(mixture.Mixture.ModelArms) != 2 ||
		!reflect.DeepEqual(mixture.Mixture.Decisions[0].ArmIDs, sortedStrings(general.ID, premium.ID)) {
		t.Fatalf("candidate iteration did not bind effective adapters: %#v", mixture.Mixture)
	}
}

func TestLoRAFullPricingAndTopologyDriftStayOrthogonal(t *testing.T) {
	base := mustLoRAMixture(t, loraSnapshotTestYAML)
	general := requireSnapshotArm(t, base, "general-expert")

	cachePriceYAML := strings.Replace(loraSnapshotTestYAML, "cached_input_per_1m: 0.05", "cached_input_per_1m: 0.06", 1)
	cachePrice := mustLoRAMixture(t, cachePriceYAML)
	changedGeneral := requireSnapshotArm(t, cachePrice, "general-expert")
	if general.InputCostPerMillionTokensUSD != changedGeneral.InputCostPerMillionTokensUSD ||
		general.OutputCostPerMillionTokensUSD != changedGeneral.OutputCostPerMillionTokensUSD ||
		*general.ConfigDigest == *changedGeneral.ConfigDigest ||
		base.Mixture.PoolDigest == cachePrice.Mixture.PoolDigest ||
		base.BackendTopologyDigest != cachePrice.BackendTopologyDigest {
		t.Fatalf("full pricing drift was not pool-owned: base=%#v changed=%#v", general, changedGeneral)
	}
	assertNonPoolFactorsEqual(t, base, cachePrice)

	endpointYAML := strings.Replace(loraSnapshotTestYAML, "foundation.private.test:8000", "replacement.private.test:9000", 1)
	endpoint := mustLoRAMixture(t, endpointYAML)
	if base.BackendTopologyDigest == endpoint.BackendTopologyDigest ||
		base.Mixture.PoolDigest != endpoint.Mixture.PoolDigest ||
		!reflect.DeepEqual(base.Mixture.ModelArms, endpoint.Mixture.ModelArms) {
		t.Fatalf("base endpoint drift was not topology-owned: base=%#v changed=%#v", base, endpoint)
	}
	assertNonPoolFactorsEqual(t, base, endpoint)
}

func TestLoRAExplicitFreeCacheWriteRemainsPartOfPricingIdentity(t *testing.T) {
	base := mustLoRAMixture(t, loraSnapshotTestYAML)
	premium := requireSnapshotArm(t, base, "premium-expert")
	explicitFreeYAML := strings.Replace(
		loraSnapshotTestYAML,
		"        completion_per_1m: 3.5",
		"        completion_per_1m: 3.5\n        cache_write_per_1m: 0",
		1,
	)
	explicitFree := mustLoRAMixture(t, explicitFreeYAML)
	changedPremium := requireSnapshotArm(t, explicitFree, "premium-expert")
	if premium.InputCostPerMillionTokensUSD != changedPremium.InputCostPerMillionTokensUSD ||
		premium.OutputCostPerMillionTokensUSD != changedPremium.OutputCostPerMillionTokensUSD ||
		*premium.ConfigDigest == *changedPremium.ConfigDigest ||
		base.Mixture.PoolDigest == explicitFree.Mixture.PoolDigest ||
		base.BackendTopologyDigest != explicitFree.BackendTopologyDigest {
		t.Fatalf("explicit free cache-write rate was not frozen: base=%#v changed=%#v", premium, changedPremium)
	}
	assertNonPoolFactorsEqual(t, base, explicitFree)
}

func TestLoRAAliasWithDirectBackendFailsClosed(t *testing.T) {
	directBackendYAML := strings.Replace(
		loraSnapshotTestYAML,
		"    - name: premium-expert\n      pricing:",
		"    - name: premium-expert\n"+
			"      backend_refs:\n"+
			"        - {endpoint: premium.private.test:8100}\n"+
			"      pricing:",
		1,
	)
	snapshot, err := ModelArmSnapshotFromYAML([]byte(directBackendYAML), "revision")
	if err != nil {
		t.Fatalf("direct alias backend snapshot: %v", err)
	}
	if mixture := requireSingleMixture(t, snapshot); mixture.Ready {
		t.Fatalf("ambiguous direct LoRA transport was exposed as ready: %#v", mixture)
	}
}

func TestLoRANonUSDPricingFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{
			name: "inherited base pricing",
			yaml: strings.Replace(
				loraSnapshotTestYAML, "      pricing:\n        prompt_per_1m: 0.25", "      pricing:\n        currency: EUR\n        prompt_per_1m: 0.25", 1,
			),
		},
		{
			name: "explicit adapter override",
			yaml: strings.Replace(
				loraSnapshotTestYAML, "    - name: premium-expert\n      pricing:", "    - name: premium-expert\n      pricing:\n        currency: EUR", 1,
			),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot, err := ModelArmSnapshotFromYAML([]byte(test.yaml), "revision")
			if err != nil {
				t.Fatalf("non-USD snapshot parse: %v", err)
			}
			if mixture := requireSingleMixture(t, snapshot); mixture.Ready {
				t.Fatalf("non-USD LoRA pricing was exposed as ready: %#v", mixture)
			}
		})
	}
}

func TestLoRATopologyAndPublicSnapshotDoNotLeakBaseDetails(t *testing.T) {
	mixture := mustLoRAMixture(t, loraSnapshotTestYAML)
	assertLoRATopologyUsesOnlyBase(t, loraSnapshotTestYAML, mixture)
	encoded, err := json.Marshal(mixture.Mixture)
	if err != nil {
		t.Fatalf("marshal LoRA mixture: %v", err)
	}
	for _, privateValue := range []string{
		"foundation-private", "PrivateCorp/Foundation-Runtime", "foundation.private.test", "base-secret",
	} {
		if strings.Contains(string(encoded), privateValue) {
			t.Fatalf("public LoRA snapshot leaked base detail %q: %s", privateValue, encoded)
		}
	}
}

func mustLoRAMixture(t *testing.T, data string) MixtureTargetSnapshot {
	t.Helper()
	snapshot, err := ModelArmSnapshotFromYAML([]byte(data), "revision")
	if err != nil {
		t.Fatalf("LoRA snapshot: %v", err)
	}
	mixture := requireSingleMixture(t, snapshot)
	if !mixture.Ready {
		t.Fatalf("LoRA mixture is unavailable: %#v", mixture)
	}
	return mixture
}

func requireSnapshotArm(t *testing.T, mixture MixtureTargetSnapshot, model string) ModelArm {
	t.Helper()
	for _, arm := range mixture.Mixture.ModelArms {
		if arm.Model == model {
			return arm
		}
	}
	t.Fatalf("arm %q is unavailable in %#v", model, mixture.Mixture.ModelArms)
	return ModelArm{}
}

func sortedStrings(values ...string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
}

func assertLoRATopologyUsesOnlyBase(t *testing.T, data string, mixture MixtureTargetSnapshot) {
	t.Helper()
	cfg, err := routerconfig.ParseYAMLBytes([]byte(data))
	if err != nil {
		t.Fatalf("parse topology fixture: %v", err)
	}
	canonical := routerconfig.CanonicalConfigFromRouterConfig(cfg)
	want := backendTopologyDigestForModels(canonical, map[string]struct{}{"foundation-private": {}})
	if mixture.BackendTopologyDigest != want || !digestPattern.MatchString(want) {
		t.Fatalf("LoRA topology was not deduplicated by base: got=%q want=%q", mixture.BackendTopologyDigest, want)
	}
}

func assertNonPoolFactorsEqual(t *testing.T, left, right MixtureTargetSnapshot) {
	t.Helper()
	if left.Mixture.RecipeDigest != right.Mixture.RecipeDigest ||
		left.Mixture.BindingDigest != right.Mixture.BindingDigest ||
		left.Mixture.SelectorDigest != right.Mixture.SelectorDigest ||
		left.Mixture.AdaptationDigest != right.Mixture.AdaptationDigest {
		t.Fatalf("pool/topology drift crossed policy factors: left=%#v right=%#v", left, right)
	}
}
