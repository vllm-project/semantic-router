package config

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestCompileManagedRoutingSnapshotBuildsEntrypointRuntimeWithoutProviderRoutes(t *testing.T) {
	base := DefaultGlobalConfig()
	configureValidManagedAccess(&base)
	snapshot := compileManagedSnapshotFixture(t)

	compiled, err := CompileManagedRoutingSnapshot(&base, snapshot)
	if err != nil {
		t.Fatalf("CompileManagedRoutingSnapshot() error = %v", err)
	}
	if compiled.DocumentHash != snapshot.Digest || compiled.ControlPlane.Mode != ControlPlaneModeManaged {
		t.Fatalf("compiled identity = hash %q mode %q", compiled.DocumentHash, compiled.ControlPlane.Mode)
	}
	if len(compiled.VLLMEndpoints) != 0 {
		t.Fatalf("managed snapshot created source provider endpoints: %+v", compiled.VLLMEndpoints)
	}
	model, exists := compiled.ModelConfig["local/fast"]
	if !exists || model.ResourceID != "mdl_fast" || model.ResourceRevision != 3 ||
		model.Execution.RequestTimeout != "30s" || model.Execution.StreamTimeout != "2m" ||
		!reflect.DeepEqual(model.Aliases, []string{"fast-upstream"}) ||
		model.Reasoning.Type != ReasoningFamilyTypeReasoningEffort ||
		!reflect.DeepEqual(model.Reasoning.Efforts, []string{"high", "medium"}) {
		t.Fatalf("compiled Model = %+v", model)
	}
	if model.RuntimePricing.InputCostPerMillionTokens == nil ||
		*model.RuntimePricing.InputCostPerMillionTokens != "0.25" {
		t.Fatalf("compiled Model pricing = %+v", model.RuntimePricing)
	}
	resolved, err := compiled.ResolveEntrypoint(
		"vllm-sr/blend", "/v1/chat/completions",
		map[string]EntrypointClaimValue{"routing_tier": {Kind: "string", String: "free"}},
	)
	if err != nil || resolved.Outcome != EntrypointResolveMatched || resolved.Recipe == nil {
		t.Fatalf("ResolveEntrypoint() = %+v, %v", resolved, err)
	}
	decisions := resolved.Recipe.Profile.Decisions
	if len(decisions) != 1 || len(decisions[0].ModelRefs) != 1 || decisions[0].ModelRefs[0].Model != "local/fast" {
		t.Fatalf("derived Recipe decisions = %+v", decisions)
	}
	if base.DocumentHash == snapshot.Digest || len(base.Entrypoints) != 0 {
		t.Fatal("CompileManagedRoutingSnapshot mutated bootstrap config")
	}

	authoring := CanonicalConfigFromRouterConfig(compiled)
	if len(authoring.Models) != 0 || len(authoring.Recipes) != 0 || len(authoring.Entrypoints) != 0 {
		t.Fatalf("managed publication leaked into authoring export: %+v", authoring)
	}
	exported, err := yaml.Marshal(authoring)
	if err != nil {
		t.Fatal(err)
	}
	for _, compilerState := range []string{"provider-credential-fast", "mdl_fast", "backends:"} {
		if strings.Contains(string(exported), compilerState) {
			t.Fatalf("managed authoring export contains %q:\n%s", compilerState, exported)
		}
	}
}

func TestCompileManagedRoutingSnapshotRejectsUnverifiedAndDivergentRecipeState(t *testing.T) {
	base := DefaultGlobalConfig()
	configureValidManagedAccess(&base)

	t.Run("digest", func(t *testing.T) {
		snapshot := compileManagedSnapshotFixture(t)
		snapshot.Digest = strings.Repeat("0", 64)
		if _, err := CompileManagedRoutingSnapshot(&base, snapshot); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Fatalf("CompileManagedRoutingSnapshot() error = %v", err)
		}
	})

	t.Run("decision metadata", func(t *testing.T) {
		bundle := managedSnapshotBundleFixture()
		bundle.Recipes[0].Decisions[0].Name = "Different"
		snapshot, err := routingsnapshot.Compile(bundle)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := CompileManagedRoutingSnapshot(&base, snapshot); err == nil || !strings.Contains(err.Error(), "decision metadata") {
			t.Fatalf("CompileManagedRoutingSnapshot() error = %v", err)
		}
	})

	t.Run("detached routing wrapper", func(t *testing.T) {
		bundle := managedSnapshotBundleFixture()
		bundle.Recipes[0].Document = json.RawMessage(`{"routing":{"decisions":[{"id":"dec_simple","name":"Simple","rules":{}}]}}`)
		snapshot, err := routingsnapshot.Compile(bundle)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := CompileManagedRoutingSnapshot(&base, snapshot); err == nil || !strings.Contains(err.Error(), "unsupported field") {
			t.Fatalf("CompileManagedRoutingSnapshot() error = %v", err)
		}
	})
}

func TestParseManagedRecipeDocumentRejectsUnknownNestedFields(t *testing.T) {
	document := json.RawMessage(`{
  "signals": {},
  "projections": {},
  "decisions": [{"name":"Simple","rules":{},"technicalExplanation":"leak"}]
}`)
	if _, _, err := ParseManagedRecipeDocument(document); err == nil {
		t.Fatal("ParseManagedRecipeDocument accepted an unknown nested field")
	}
}

func TestParseManagedRecipeDocumentRejectsCompilerOwnedDecisionIdentity(t *testing.T) {
	document := json.RawMessage(`{"decisions":[{"id":"dec_simple","name":"Simple","rules":{}}]}`)
	if _, _, err := ParseManagedRecipeDocument(document); err == nil || !strings.Contains(err.Error(), "compiler-owned") {
		t.Fatalf("ParseManagedRecipeDocument() error = %v", err)
	}
}

func TestParseManagedRecipeDocumentAcceptsEmptyControlPlaneDraft(t *testing.T) {
	document := json.RawMessage(`{"signals":{},"projections":{},"decisions":[]}`)
	parsed, canonical, err := ParseManagedRecipeDocument(document)
	if err != nil {
		t.Fatalf("ParseManagedRecipeDocument() error = %v", err)
	}
	if len(parsed.Decisions) != 0 || !json.Valid(canonical) {
		t.Fatalf("empty draft = %+v, %s", parsed, canonical)
	}
}

func TestValidateManagedRoutingSnapshotRejectsSemanticRecipeErrors(t *testing.T) {
	bundle := managedSnapshotBundleFixture()
	bundle.Recipes[0].Document = json.RawMessage(`{
  "signals": {},
  "projections": {},
  "decisions": [{
    "name":"Simple",
    "rules":{"type":"metadata","name":"missing-metadata-signal"}
  }]
}`)
	snapshot, err := routingsnapshot.Compile(bundle)
	if err != nil {
		t.Fatalf("routingsnapshot.Compile() error = %v", err)
	}
	if err := ValidateManagedRoutingSnapshot(snapshot); err == nil ||
		!strings.Contains(err.Error(), "signal") || !strings.Contains(err.Error(), "missing-metadata-signal") {
		t.Fatalf("ValidateManagedRoutingSnapshot() error = %v", err)
	}
}

func TestRuntimeAssignmentsFromSnapshotPreserveFallbackContract(t *testing.T) {
	assignments := map[string]routingsnapshot.AssignmentSet{
		"dec_simple": {
			Models: []routingsnapshot.Assignment{
				{ModelID: "mdl_fast", ModelRevision: 3, Priority: 0, Weight: "1"},
				{ModelID: "mdl_backup", ModelRevision: 7, Priority: 1, Weight: "1"},
			},
			Fallback: &routingsnapshot.FallbackPolicy{Strategy: "priority", On: []string{"unavailable", "timeout"}},
		},
	}

	runtimeAssignments, err := runtimeAssignmentsFromSnapshot(
		assignments,
		map[string]routingsnapshot.Model{
			"mdl_fast":   {ID: "mdl_fast", Revision: 3, Name: "fast"},
			"mdl_backup": {ID: "mdl_backup", Revision: 7, Name: "backup"},
		},
	)
	if err != nil {
		t.Fatalf("runtimeAssignmentsFromSnapshot() error = %v", err)
	}
	set := runtimeAssignments["dec_simple"]
	if len(set.Models) != 2 || set.Models[0].Priority != 0 || set.Models[1].Priority != 1 {
		t.Fatalf("canonical Models = %+v", set.Models)
	}
	if set.Fallback == nil || set.Fallback.Strategy != "priority" ||
		!reflect.DeepEqual(set.Fallback.On, []string{"unavailable", "timeout"}) {
		t.Fatalf("canonical Fallback = %+v", set.Fallback)
	}
}

func compileManagedSnapshotFixture(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(managedSnapshotBundleFixture())
	if err != nil {
		t.Fatalf("routingsnapshot.Compile() error = %v", err)
	}
	return snapshot
}

func managedSnapshotBundleFixture() routingsnapshot.Bundle {
	inputCost := "0.25"
	return routingsnapshot.Bundle{
		NamespaceID: "11111111-1111-4111-8111-111111111111", Revision: 9, Currency: "USD",
		Models: []routingsnapshot.Model{{
			ID: "mdl_fast", Revision: 3,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "local/fast", Aliases: []string{"fast-upstream"},
			Capabilities: []string{"chat", "tools"},
			Reasoning: routingsnapshot.ReasoningFamily{
				Type: ReasoningFamilyTypeReasoningEffort, Efforts: []string{"medium", "high"},
			},
			Execution: routingsnapshot.ModelExecution{MaxRetries: 2, RequestTimeout: "30s", StreamTimeout: "2m"},
			Pricing:   routingsnapshot.ModelPricing{InputCostPerMillionTokens: &inputCost},
			Backends: []routingsnapshot.Backend{{
				ID: "be_fast", ProviderID: "provider_local", WireFormat: "openai.chat.v1",
				Origin: "http://models.example:8000", ProviderModelID: "fast",
				ProviderCredentialID: "provider-credential-fast",
				Connection:           routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "rcp_balance", Revision: 5, Name: "balance",
			Decisions: []routingsnapshot.Decision{{ID: "dec_simple", Name: "Simple", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
			Document: json.RawMessage(`{
  "signals": {},
  "projections": {},
  "decisions": [{"name":"Simple","rules":{}}]
}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "ep_blend", Revision: 7, Name: "blend", Aliases: []string{"vllm-sr/blend"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule_free", Name: "free", RecipeID: "rcp_balance", RecipeRevision: 5,
				Matchers: []routingsnapshot.Matcher{{Claim: &routingsnapshot.ClaimMatcher{
					Name: "routing_tier", Value: routingsnapshot.ClaimValue{Kind: "string", String: "free"},
				}}},
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"dec_simple": {Models: []routingsnapshot.Assignment{{ModelID: "mdl_fast", ModelRevision: 3, Weight: "1"}}},
				},
			}},
		}},
	}
}
