package config

import (
	"math"
	"strings"
	"testing"
)

func TestNativeCascadeAndCalibrationUseCommonContract(t *testing.T) {
	digest := strings.Repeat("a", 64)
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	algorithm := findRecipe(cfg.Recipes, "native-decisions").Profile.Decisions[0].Algorithm
	if algorithm.Budget.MaxCalls != 4 || len(algorithm.Stages) != 3 || algorithm.Quality.Type != "uncalibrated" {
		t.Fatalf("cascade contract lost: %+v", algorithm)
	}
	// Configuration parsing checks references and immutable identities without
	// opening local files. The execution owner verifies the artifact bytes.
	canonical := CanonicalConfigFromRouterConfig(cfg)
	risk := 0.01
	canonical.Evaluation = &CanonicalEvaluation{Calibrations: []CalibrationArtifact{{Name: "suite-v1", Source: "./calibration/suite.json", SHA256: digest}}}
	canonical.Recipes[0].Routing.Decisions[0].Algorithm.Quality = &NativeQualityConfig{
		Type: "calibrated", Calibration: "suite-v1", Loss: "bundle_error", MaxRisk: &risk,
	}
	calibrated, err := normalizeCanonicalConfig(&canonical)
	if err != nil {
		t.Fatal(err)
	}
	artifact, ok := calibrated.Calibration("suite-v1")
	if !ok || artifact.SHA256 != digest {
		t.Fatalf("calibration reference = %+v, %v", artifact, ok)
	}
	canonical.Recipes[0].Routing.Decisions[0].Algorithm.Quality.Calibration = "missing"
	if _, err := normalizeCanonicalConfig(&canonical); err == nil || !strings.Contains(err.Error(), "unknown evaluation calibration") {
		t.Fatalf("unresolved evidence reference accepted: %v", err)
	}
}

func TestNativeArtifactAndQualityValidation(t *testing.T) {
	digest := strings.Repeat("f", 64)
	for _, test := range []struct{ source, hash string }{
		{"https://example.test/policy.json", digest},
		{"", digest},
		{"policy.json", "not-a-hash"},
		{"policy.json", strings.Repeat("z", 64)},
	} {
		if err := validateNativeArtifact(test.source, test.hash); err == nil {
			t.Fatalf("invalid artifact accepted: %q, %q", test.source, test.hash)
		}
	}
	for _, risk := range []float64{math.NaN(), math.Inf(1), -0.01, 1.01} {
		if err := validateNativeQuality(&NativeQualityConfig{Type: "calibrated", Calibration: "suite", Loss: "bundle_error", MaxRisk: &risk}); err == nil {
			t.Fatalf("invalid risk accepted: %v", risk)
		}
	}
	if err := validateNativeQuality(&NativeQualityConfig{Type: "uncalibrated", Calibration: "suite"}); err == nil {
		t.Fatal("uncalibrated policy accepted calibrated evidence")
	}
	if err := validateNativeQuality(&NativeQualityConfig{Type: "calibrated", Calibration: "suite", Loss: "bundle_error"}); err == nil {
		t.Fatal("omitted risk target interpreted as a scientific default")
	}
}

func TestNativeDecisionsHaveIndependentAlgorithmBudgets(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	first := canonical.Recipes[0].Routing.Decisions[0]
	second := copyDecisions([]Decision{first})[0]
	second.Name = "alternate"
	second.Algorithm.Budget = &AlgorithmBudget{Deadline: "9s", MaxCalls: 2}
	canonical.Recipes[0].Routing.Decisions = append(canonical.Recipes[0].Routing.Decisions, second)
	compiled, err := normalizeCanonicalConfig(&canonical)
	if err != nil {
		t.Fatal(err)
	}
	native := findRecipe(compiled.Recipes, "native-decisions")
	if native.Profile.Decisions[0].Algorithm.Budget.MaxCalls != 4 || native.Profile.Decisions[1].Algorithm.Budget.MaxCalls != 2 {
		t.Fatal("decisions shared their algorithm budgets")
	}
	exported := CanonicalConfigFromRouterConfig(compiled)
	exported.Recipes[0].Routing.Decisions[1].Algorithm.Budget.MaxCalls = 99
	if native.Profile.Decisions[1].Algorithm.Budget.MaxCalls != 2 {
		t.Fatal("export mutated retained algorithm budget")
	}
	if !compiled.IsRecipeReachableForRouting(native.Name) {
		t.Fatal("published native signal consumers are not reachable")
	}
	compiled.Listeners[0].SystemOne = nil
	if compiled.IsRecipeReachableForRouting(native.Name) {
		t.Fatal("unpublished recipe remains reachable")
	}
}

func TestNativeStageRestrictionsAndSharedCandidateRoster(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	decision := findRecipe(cfg.Recipes, "native-decisions").Profile.Decisions[0]
	decision.Algorithm.Stages[1].Model = "undeclared"
	if err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm); err == nil {
		t.Fatal("stage expanded the sole candidate roster")
	}
}

func TestNativeExportDoesNotMutateRetainedPolicy(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	retained := findRecipe(cfg.Recipes, "native-decisions").Profile.Decisions[0].Algorithm
	exported := CanonicalConfigFromRouterConfig(cfg).Recipes[0].Routing.Decisions[0].Algorithm
	exported.Stages[0].Model = "different"
	*exported.Stages[2].Enabled = true
	*exported.Quality.Acceptance.Rules[0].State = "another-state"
	*exported.Quality.Acceptance.Rules[0].Predicate.GTE = 0.01
	if retained.Stages[0].Model != "kai" || retained.Stages[2].IsEnabled() ||
		*retained.Quality.Acceptance.Rules[0].State != "document" || *retained.Quality.Acceptance.Rules[0].Predicate.GTE != 0.8 {
		t.Fatal("export mutated a retained native execution graph")
	}
}

func TestNativeAcceptanceProbabilityBounds(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), -0.01, 1.01} {
		err := validateNativeAcceptance(&NativeAcceptance{Rules: []NativeAcceptanceRule{{
			QuestionType: "choice", Field: "top_probability", Predicate: NumericPredicate{GTE: &value},
		}}})
		if err == nil {
			t.Fatalf("invalid probability threshold accepted: %v", value)
		}
	}
	for _, value := range []float64{0, 0.9, 1} {
		err := validateNativeAcceptance(&NativeAcceptance{Rules: []NativeAcceptanceRule{{
			QuestionType: "noul", Field: "top_probability", Predicate: NumericPredicate{GTE: &value},
		}}})
		if err != nil {
			t.Fatalf("valid native threshold failed: %v", err)
		}
	}
}

func TestChatRecipeDoesNotSilentlyIgnoreNativeBudget(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "entrypoints:\n", "routing:\n  budget: {deadline: 3s, max_calls: 4}\nentrypoints:\n", 1)
	_, err := ParseYAMLBytes([]byte(raw))
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("Chat routing accepted an unenforced native budget: %v", err)
	}
}
