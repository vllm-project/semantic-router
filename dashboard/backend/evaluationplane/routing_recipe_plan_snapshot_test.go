package evaluationplane

import (
	"reflect"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRoutingRecipePlanFromSnapshotFreezesOnlyReachableDecisionGraph(t *testing.T) {
	mixture := brokerTestMixture()
	routing := routerconfig.CanonicalRouting{
		Decisions: []routerconfig.Decision{
			{
				Name: "route",
				Rules: routerconfig.RuleNode{Operator: "AND", Conditions: []routerconfig.RuleNode{
					{Type: "KEYWORD", Name: "alpha"},
					{Type: routerconfig.SignalTypeClassifier, Name: "risk", Label: "RISKY"},
					{Type: routerconfig.SignalTypeProjection, Name: "final-route"},
				}},
			},
			{Name: "secondary", Rules: routerconfig.RuleNode{Operator: "OR", Conditions: []routerconfig.RuleNode{
				{Type: routerconfig.SignalTypeKeyword, Name: "alpha"},
				{Type: routerconfig.SignalTypeLanguage, Name: "en"},
			}}},
		},
		Projections: routerconfig.CanonicalProjections{
			Scores: []routerconfig.ProjectionScore{
				{
					Name: "base-score", Method: "weighted_sum",
					Inputs: []routerconfig.ProjectionScoreInput{
						{Type: routerconfig.SignalTypeContext, Name: "turns", Weight: 1},
						{Type: routerconfig.ProjectionInputKBMetric, KB: "docs", Metric: "best_score", Weight: 1},
					},
				},
				{
					Name: "semantic-score", Method: "weighted_sum",
					Inputs: []routerconfig.ProjectionScoreInput{
						{Type: routerconfig.SignalTypeEmbedding, Name: "semantic", Weight: 1},
					},
				},
				{
					Name: "final-score", Method: "weighted_sum",
					Inputs: []routerconfig.ProjectionScoreInput{
						{Type: routerconfig.SignalTypeProjection, Name: "base-band", ValueSource: routerconfig.ProjectionValueSourceConfidence, Weight: 1},
						{Type: routerconfig.SignalTypeProjection, Name: "semantic-score", ValueSource: routerconfig.ProjectionValueSourceScore, Weight: 1},
					},
				},
				{
					Name: "unreachable-score", Method: "weighted_sum",
					Inputs: []routerconfig.ProjectionScoreInput{
						{Type: routerconfig.SignalTypeStructure, Name: "unused", Weight: 1},
					},
				},
			},
			Mappings: []routerconfig.ProjectionMapping{
				{Name: "base", Source: "base-score", Outputs: []routerconfig.ProjectionMappingOutput{{Name: "base-band"}}},
				{Name: "final", Source: "final-score", Outputs: []routerconfig.ProjectionMappingOutput{{Name: "final-route"}}},
				{Name: "unused", Source: "unreachable-score", Outputs: []routerconfig.ProjectionMappingOutput{{Name: "unused-output"}}},
			},
		},
	}

	plan, err := routingRecipePlanFromSnapshot(routing, *mixture)
	if err != nil {
		t.Fatalf("derive routing recipe plan: %v", err)
	}
	wantSignals := []RoutingRecipeInputSpec{
		{ID: "classifier:risk:RISKY", ValueKind: "numeric"},
		{ID: "context:turns", ValueKind: "numeric"},
		{ID: "embedding:semantic", ValueKind: "numeric"},
		{ID: "kb_metric:docs:best_score", ValueKind: "numeric"},
		{ID: "keyword:alpha", ValueKind: "numeric"},
		{ID: "language:en", ValueKind: "numeric"},
	}
	wantProjections := []RoutingRecipeProjectionSpec{
		{ID: "projection:base-band", ValueKind: "probability", OutcomeBinding: "selected_is_oracle"},
		{ID: "projection:final-route", ValueKind: "probability", OutcomeBinding: "selected_is_oracle"},
	}
	if !reflect.DeepEqual(plan.Signals, wantSignals) || !reflect.DeepEqual(plan.Projections, wantProjections) {
		t.Fatalf("reachable plan graph = signals %#v projections %#v", plan.Signals, plan.Projections)
	}
	if !reflect.DeepEqual(plan.TopK, []int{1, 2}) || !reflect.DeepEqual(plan.ArmIDs, []string{"arm-fast", "arm-strong"}) {
		t.Fatalf("frozen pool planning = arms %v top-k %v", plan.ArmIDs, plan.TopK)
	}
	wantTargetDigest, digestErr := routingRecipeTargetSnapshotDigest(*mixture)
	if digestErr != nil || plan.TargetSnapshotDigest != wantTargetDigest {
		t.Fatalf("target snapshot digest = %q want %q err=%v", plan.TargetSnapshotDigest, wantTargetDigest, digestErr)
	}
	if err := ValidateRoutingRecipePlan(plan); err != nil {
		t.Fatalf("derived plan is invalid: %v", err)
	}
}

func TestRoutingRecipeTopKIsUniqueSortedAndPoolBounded(t *testing.T) {
	tests := map[int][]int{
		1: {1},
		2: {1, 2},
		3: {1, 3},
		4: {1, 3, 4},
		5: {1, 3, 5},
		8: {1, 3, 5},
	}
	for armCount, want := range tests {
		if got := routingRecipeTopK(armCount); !reflect.DeepEqual(got, want) {
			t.Fatalf("routingRecipeTopK(%d) = %v, want %v", armCount, got, want)
		}
	}
}

func TestMixtureRoutingRecipePlanIsRequiredCopiedAndDigestBound(t *testing.T) {
	mixture := brokerTestMixture()
	if err := validateManifestMixtureContract(mixture); err != nil {
		t.Fatalf("valid frozen mixture rejected: %v", err)
	}

	catalog := catalogMixtureFromManifest(mixture)
	roundTrip := manifestMixtureFromCatalog(catalog)
	if !reflect.DeepEqual(roundTrip, mixture) {
		t.Fatalf("catalog/manifest routing plan round trip drifted:\n got %#v\nwant %#v", roundTrip, mixture)
	}
	catalog.RoutingRecipePlan.ArmIDs[0] = "mutated"
	if mixture.RoutingRecipePlan.ArmIDs[0] == "mutated" {
		t.Fatal("catalog conversion aliased the manifest routing plan")
	}

	missing := *mixture
	missing.RoutingRecipePlan = RoutingRecipePlan{}
	if err := validateManifestMixtureContract(&missing); err == nil {
		t.Fatal("mixture without a routing recipe plan was accepted")
	}

	mutatedTarget := *mixture
	mutatedTarget.RoutingRecipePlan.TargetSnapshotDigest = digestString("forged-target")
	mutatedTarget.RoutingRecipePlan, _ = canonicalRoutingRecipePlan(mutatedTarget.RoutingRecipePlan)
	if err := validateManifestMixtureContract(&mutatedTarget); err == nil {
		t.Fatal("routing recipe plan detached from mixture component digests was accepted")
	}
	mutatedTopK := *mixture
	mutatedTopK.RoutingRecipePlan.TopK = []int{1}
	mutatedTopK.RoutingRecipePlan, _ = canonicalRoutingRecipePlan(mutatedTopK.RoutingRecipePlan)
	if err := validateManifestMixtureContract(&mutatedTopK); err == nil {
		t.Fatal("routing recipe plan with a non-derived top-k schedule was accepted")
	}
	mutatedKind := *mixture
	mutatedKind.RoutingRecipePlan.Signals = []RoutingRecipeInputSpec{{ID: "context:turns", ValueKind: "none"}}
	mutatedKind.RoutingRecipePlan, _ = canonicalRoutingRecipePlan(mutatedKind.RoutingRecipePlan)
	if err := validateManifestMixtureContract(&mutatedKind); err == nil {
		t.Fatal("routing recipe plan with a non-numeric signal was accepted")
	}

	baseValue, err := manifestMixtureCanonicalValue(mixture)
	if err != nil {
		t.Fatalf("canonicalize base mixture: %v", err)
	}
	mutatedPlan := *mixture
	mutatedPlan.RoutingRecipePlan.Signals = []RoutingRecipeInputSpec{{ID: "context:turns", ValueKind: "numeric"}}
	mutatedPlan.RoutingRecipePlan, err = canonicalRoutingRecipePlan(mutatedPlan.RoutingRecipePlan)
	if err != nil {
		t.Fatalf("canonicalize changed routing recipe plan: %v", err)
	}
	mutatedValue, err := manifestMixtureCanonicalValue(&mutatedPlan)
	if err != nil {
		t.Fatalf("canonicalize changed mixture: %v", err)
	}
	baseDigest, _ := canonicalValueDigest(baseValue)
	mutatedDigest, _ := canonicalValueDigest(mutatedValue)
	if baseDigest == mutatedDigest {
		t.Fatal("manifest mixture digest ignored its routing recipe plan")
	}
	permutedPlan := *mixture
	permutedPlan.RoutingRecipePlan.ArmIDs = []string{"arm-strong", "arm-fast"}
	permutedValue, err := manifestMixtureCanonicalValue(&permutedPlan)
	if err != nil {
		t.Fatalf("canonicalize semantically permuted routing plan: %v", err)
	}
	permutedDigest, _ := canonicalValueDigest(permutedValue)
	if permutedDigest != baseDigest {
		t.Fatal("manifest canonical digest changed for a set-only routing plan permutation")
	}
}

func TestRoutingRecipeTargetSnapshotDigestExcludesPlanAndBindsComponents(t *testing.T) {
	mixture := brokerTestMixture()
	base, err := routingRecipeTargetSnapshotDigest(*mixture)
	if err != nil {
		t.Fatalf("digest routing recipe target snapshot: %v", err)
	}
	planOnly := *mixture
	planOnly.RoutingRecipePlan.PlanDigest = digestString("plan-only-mutation")
	if got, gotErr := routingRecipeTargetSnapshotDigest(planOnly); gotErr != nil || got != base {
		t.Fatalf("plan entered its own target snapshot digest: got %q want %q err=%v", got, base, gotErr)
	}
	component := *mixture
	component.BindingDigest = digestString("changed-binding")
	if got, gotErr := routingRecipeTargetSnapshotDigest(component); gotErr != nil || got == base {
		t.Fatalf("immutable component did not change target snapshot digest: got %q base %q err=%v", got, base, gotErr)
	}
}

func TestGeneratedMixtureCarriesValidatedRoutingRecipePlan(t *testing.T) {
	snapshot, err := ModelArmSnapshotFromYAML([]byte(modelArmTestYAML), "revision")
	if err != nil {
		t.Fatalf("build model arm snapshot: %v", err)
	}
	mixture := requireSingleMixture(t, snapshot).Mixture
	if err := ValidateRoutingRecipePlan(mixture.RoutingRecipePlan); err != nil {
		t.Fatalf("server-generated mixture routing plan: %v", err)
	}
	if err := validateMixtureContract(&mixture); err != nil {
		t.Fatalf("server-generated mixture contract: %v", err)
	}
}
