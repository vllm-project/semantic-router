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

const modelArmTestYAML = `version: v0.3
providers:
  defaults:
    default_model: Org/Fast Model
  models:
    - name: Org/Fast Model
      provider_model_id: PrivateOrg/Secret-Upstream-ID
      backend_refs:
        - name: private-primary
          endpoint: private.models.example.test:8000/v1
          api_key: literal-test-secret
      pricing:
        currency: usd
        prompt_per_1m: 1.25
        completion_per_1m: 4.5
    - name: local/omni
      backend_refs:
        - name: local-primary
          endpoint: local-private.example.test:8001
    - name: metadata-only
      provider_model_id: metadata-only-upstream
      pricing:
        currency: USD
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
    - name: non-usd
      provider_model_id: foreign-upstream
      backend_refs:
        - endpoint: foreign-private.example.test:8002
      pricing:
        currency: EUR
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
    - name: ambiguous-unpriced
      backend_refs:
        - endpoint: ambiguous-private.example.test:8003
      pricing:
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
routing:
  modelCards:
    - name: Org/Fast Model
      param_size: 7B
      context_window_size: 32768
      capabilities: [vision, chat, vision, ""]
      modality: ar
    - name: local/omni
      capabilities: [omni, audio, document_understanding, video_understanding]
      modality: omni
    - name: metadata-only
      modality: text
    - name: non-usd
      modality: text
    - name: ambiguous-unpriced
      modality: text
`

func TestModelArmSnapshotFromYAMLIsCanonicalDeterministicAndConnectivityFree(t *testing.T) {
	runtimeRevision := "git-test-revision"
	snapshot, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), runtimeRevision)
	if err != nil {
		t.Fatalf("ModelArmSnapshotFromYAML() error = %v", err)
	}
	if snapshot.ConfigDigest != digestBytes([]byte(modelArmTestYAML)) {
		t.Fatalf("config digest = %q, want digest of the parsed bytes", snapshot.ConfigDigest)
	}
	if !digestPattern.MatchString(snapshot.PolicySnapshotDigest) {
		t.Fatalf("policy snapshot digest = %q, want a sha256 digest", snapshot.PolicySnapshotDigest)
	}
	if len(snapshot.ModelArms) != 2 {
		t.Fatalf("model arms = %#v, want only the two executable, USD-compatible providers", snapshot.ModelArms)
	}

	fast := snapshot.ModelArms[0]
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
	if !digestPattern.MatchString(snapshot.BackendTopologyDigest) {
		t.Fatalf("backend topology digest = %q", snapshot.BackendTopologyDigest)
	}

	omni := snapshot.ModelArms[1]
	if omni.Model != "local/omni" {
		t.Fatalf("second model = %q, want local/omni", omni.Model)
	}
	if omni.ProviderModelIDDigest != digestString("local/omni") {
		t.Fatalf("fallback provider digest = %q, want digest of logical model", omni.ProviderModelIDDigest)
	}
	if !reflect.DeepEqual(omni.Modalities, []string{"text", "image", "document", "audio", "video"}) {
		t.Fatalf("omni modalities = %#v", omni.Modalities)
	}

	encoded, err := json.Marshal(snapshot.ModelArms)
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
	if snapshot.ConfigDigest != digestBytes([]byte(modelArmTestYAML)) || len(snapshot.ModelArms) != 2 {
		t.Fatalf("snapshot = %#v, want digest and arms from file bytes", snapshot)
	}

	absent, err := LoadModelArmSnapshot("  ", "revision")
	if err != nil {
		t.Fatalf("empty config path error = %v", err)
	}
	if absent.ConfigDigest != unavailableConfigDigest || !digestPattern.MatchString(absent.PolicySnapshotDigest) ||
		absent.BackendTopologyDigest != "" || len(absent.ModelArms) != 0 {
		t.Fatalf("empty config snapshot = %#v", absent)
	}
}

func TestSnapshotSeparatesPolicyArmAndBackendTopologyOwnership(t *testing.T) {
	base, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision-a")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}
	policyYAML := strings.Replace(modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
	policy, err := ModelArmSnapshotFromYAML([]byte(policyYAML), "revision-a")
	if err != nil {
		t.Fatalf("policy snapshot: %v", err)
	}
	if base.ConfigDigest == policy.ConfigDigest || base.PolicySnapshotDigest == policy.PolicySnapshotDigest ||
		base.BackendTopologyDigest != policy.BackendTopologyDigest || !reflect.DeepEqual(base.ModelArms, policy.ModelArms) {
		t.Fatalf("policy-only change crossed ownership: base=%#v policy=%#v", base, policy)
	}

	endpointYAML := strings.Replace(modelArmTestYAML, "private.models.example.test:8000", "replacement.models.example.test:9000", 1)
	endpoint, err := ModelArmSnapshotFromYAML([]byte(endpointYAML), "revision-a")
	if err != nil {
		t.Fatalf("endpoint snapshot: %v", err)
	}
	if base.BackendTopologyDigest == endpoint.BackendTopologyDigest || base.PolicySnapshotDigest != endpoint.PolicySnapshotDigest ||
		!reflect.DeepEqual(base.ModelArms, endpoint.ModelArms) {
		t.Fatalf("endpoint-only change must affect only topology identity: base=%#v endpoint=%#v", base, endpoint)
	}

	priceYAML := strings.Replace(modelArmTestYAML, "prompt_per_1m: 1.25", "prompt_per_1m: 1.50", 1)
	price, err := ModelArmSnapshotFromYAML([]byte(priceYAML), "revision-a")
	if err != nil {
		t.Fatalf("price snapshot: %v", err)
	}
	if base.PolicySnapshotDigest != price.PolicySnapshotDigest || base.BackendTopologyDigest != price.BackendTopologyDigest ||
		*base.ModelArms[0].ConfigDigest == *price.ModelArms[0].ConfigDigest ||
		digestJSON(base.ModelArms) == digestJSON(price.ModelArms) {
		t.Fatalf("arm price change did not alter only pool semantics: base=%#v price=%#v", base, price)
	}

	capabilityYAML := strings.Replace(modelArmTestYAML, "capabilities: [vision, chat, vision, \"\"]", "capabilities: [vision, chat, reasoning]", 1)
	capability, err := ModelArmSnapshotFromYAML([]byte(capabilityYAML), "revision-a")
	if err != nil {
		t.Fatalf("capability snapshot: %v", err)
	}
	if base.PolicySnapshotDigest != capability.PolicySnapshotDigest || base.BackendTopologyDigest != capability.BackendTopologyDigest ||
		*base.ModelArms[0].ConfigDigest == *capability.ModelArms[0].ConfigDigest ||
		digestJSON(base.ModelArms) == digestJSON(capability.ModelArms) {
		t.Fatalf("arm capability change did not alter only pool semantics: base=%#v capability=%#v", base, capability)
	}

	revision, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision-b")
	if err != nil {
		t.Fatalf("revision snapshot: %v", err)
	}
	if base.PolicySnapshotDigest != revision.PolicySnapshotDigest ||
		*base.ModelArms[0].ConfigDigest != *revision.ModelArms[0].ConfigDigest ||
		digestJSON(base.ModelArms) == digestJSON(revision.ModelArms) {
		t.Fatalf("runtime revision must alter pool snapshot without changing arm config digest")
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
	if base.ConfigDigest == commented.ConfigDigest || base.PolicySnapshotDigest != commented.PolicySnapshotDigest {
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
			name: "unknown non-zero currency",
			provider: routerconfig.CanonicalProviderModel{
				Name: "ambiguous", BackendRefs: backend,
				Pricing: routerconfig.ModelPricing{PromptPer1M: 1},
			},
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
			if got := validProviderArm(test.provider, strings.TrimSpace(test.provider.Name)); got != test.want {
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
