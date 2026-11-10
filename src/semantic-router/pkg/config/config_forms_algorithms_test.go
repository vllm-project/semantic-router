package config

import (
	"strings"
	"testing"
)

// automix valued a candidate at its quality less cost_quality_tradeoff times
// its price over $10 per 1M tokens. multi_factor rescales each factor across
// the candidates, so its weights cannot keep that tradeoff: of two candidates
// the larger weight decides. The note says so, and how to prefer a cheaper
// model within a quality margin.
func TestAutoMixMigrationSaysTheTradeoffIsNotKept(t *testing.T) {
	_, notes, err := MigrateConfig([]byte(migrateModels + `recipes:
  - name: default
    decisions:
      - name: tradeoff
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: small}, {model: large}]
        algorithm: {type: automix}
`))
	if err != nil {
		t.Fatal(err)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].decisions[tradeoff].algorithm": {
			"multi_factor cannot keep automix's cost_quality_tradeoff 0.3",
			"of two candidates the larger weight decides (quality 0.7, cost 0.3)",
			"priorities: [{factor: quality, tolerance: <margin>}, {factor: cost}]",
		},
	})
}

// The systemone selector chooses among distinct models: a selector whose
// modelRefs repeat one model, its LoRA adapters, migrates to static with a
// note, as one with a single candidate does, where the rewrite used to fail.
func TestSelectorsOverOneModelsAdaptersMigrateToStatic(t *testing.T) {
	document := strings.Replace(migrateModels, "description: Fast general model", "description: Fast general model\n      loras: [{name: math}, {name: code}]", 1) + `recipes:
  - name: default
    decisions:
`
	for _, algorithm := range []string{"router_dc", "knn", "prompt"} {
		document += "      - name: " + algorithm + "\n        rules: {operator: AND, conditions: []}\n" +
			"        modelRefs: [{model: small, lora_name: math}, {model: small, lora_name: code}]\n" +
			"        algorithm: {type: " + algorithm + "}\n"
	}
	migrated, notes, err := MigrateConfig([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(migrated), "algorithm: {type: static}"); got != 3 {
		t.Fatalf("migrate wrote\n%s\nwant three static decisions", migrated)
	}
	for _, name := range []string{"router_dc", "knn", "prompt"} {
		wantNotes(t, notes, map[string][]string{
			"recipes[default].decisions[" + name + "].algorithm": {"modelRefs repeat model small", "its first candidate serves every request"},
		})
	}
}

// A selector over two models, one of them with two LoRA adapters, keeps
// choosing between the models with the first adapter of the repeated one,
// where it used to become static and never choose the other model.
func TestSelectorOverARepeatedModelKeepsChoosing(t *testing.T) {
	migrated, notes, err := MigrateConfig([]byte(strings.Replace(migrateModels, "description: Fast general model", "description: Fast general model\n      loras: [{name: math}, {name: code}]", 1) + `recipes:
  - name: default
    decisions:
      - name: pick
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: small, lora_name: math}, {model: small, lora_name: code}, {model: large}]
        algorithm: {type: router_dc}
`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(migrated), "modelRefs: [{model: small, lora_name: math}, {model: large}]") || !strings.Contains(string(migrated), "type: systemone") {
		t.Fatalf("migrate wrote\n%s\nwant a systemone selector over small with lora_name math and large", migrated)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].decisions[pick].algorithm": {"keeps the first modelRef of each model and drops small with lora_name code"},
	})
}
