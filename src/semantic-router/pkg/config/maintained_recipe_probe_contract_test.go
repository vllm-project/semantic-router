package config

import (
	"path/filepath"
	"sort"
	"testing"
)

func assertRecipeProbeManifest(t *testing.T, name string) {
	t.Helper()
	rel := filepath.ToSlash(filepath.Join("config", "recipes", name, "probes.yaml"))
	manifest := decodeYAMLMap(t, mustReadRepoFile(t, rel), rel)
	assets := mustAssetMapValue(t, manifest, "routing_assets", rel)
	wantYAML := filepath.ToSlash(filepath.Join("config", "recipes", name, "config.yaml"))
	wantDSL := filepath.ToSlash(filepath.Join("config", "recipes", name, "recipe.dsl"))
	if assets["yaml"] != wantYAML || assets["dsl"] != wantDSL {
		t.Fatalf("%s routing_assets = %+v, want yaml=%q dsl=%q", rel, assets, wantYAML, wantDSL)
	}
	decisions, ok := manifest["decisions"].([]interface{})
	if !ok || len(decisions) == 0 {
		t.Fatalf("%s must contain a non-empty decisions list", rel)
	}

	runtimeConfig, err := ParseYAMLBytes(mustReadRepoFile(t, wantYAML))
	if err != nil {
		t.Fatalf("parse %s for probe coverage: %v", wantYAML, err)
	}
	runtimeDecisions := make([]Decision, 0)
	for _, ref := range runtimeConfig.RoutingDecisionRefs() {
		runtimeDecisions = append(runtimeDecisions, *ref.Decision)
	}
	assertProbeDecisionCoverage(t, rel, decisions, runtimeDecisions)
}

func assertProbeDecisionCoverage(
	t *testing.T,
	rel string,
	probes []interface{},
	runtimeDecisions []Decision,
) {
	t.Helper()
	decisionModels := indexDecisionModels(runtimeDecisions)
	covered := make(map[string]bool, len(probes))
	for index, rawProbe := range probes {
		expectedDecision := assertProbeCoverageEntry(
			t,
			rel,
			index,
			rawProbe,
			decisionModels,
		)
		covered[expectedDecision] = true
	}
	assertAllDecisionsCovered(t, rel, decisionModels, covered)
}

func indexDecisionModels(runtimeDecisions []Decision) map[string]map[string]bool {
	decisionModels := make(map[string]map[string]bool, len(runtimeDecisions))
	for _, decision := range runtimeDecisions {
		models := make(map[string]bool, len(decision.ModelRefs))
		for _, modelRef := range decision.ModelRefs {
			models[modelRef.Model] = true
		}
		decisionModels[decision.Name] = models
	}
	return decisionModels
}

func assertProbeCoverageEntry(
	t *testing.T,
	rel string,
	index int,
	rawProbe interface{},
	decisionModels map[string]map[string]bool,
) string {
	t.Helper()
	probe, ok := rawProbe.(map[string]interface{})
	if !ok {
		t.Fatalf("%s decisions[%d] is not a map", rel, index)
	}
	expectedDecision, ok := probe["expected_decision"].(string)
	if !ok || expectedDecision == "" {
		t.Fatalf("%s decisions[%d] must declare expected_decision", rel, index)
	}
	models, exists := decisionModels[expectedDecision]
	if !exists {
		t.Fatalf("%s references unknown decision %q", rel, expectedDecision)
	}
	expectedAlias, _ := probe["expected_alias"].(string)
	if expectedAlias != "" && !models[expectedAlias] {
		t.Fatalf(
			"%s decision %q expects alias %q outside modelRefs %v",
			rel,
			expectedDecision,
			expectedAlias,
			models,
		)
	}
	return expectedDecision
}

func assertAllDecisionsCovered(
	t *testing.T,
	rel string,
	decisionModels map[string]map[string]bool,
	covered map[string]bool,
) {
	t.Helper()
	var missing []string
	for decision := range decisionModels {
		if !covered[decision] {
			missing = append(missing, decision)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf("%s does not make every maintained decision reachable; missing %v", rel, missing)
	}
}
