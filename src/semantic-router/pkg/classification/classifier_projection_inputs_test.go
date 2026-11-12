package classification

import "testing"

// An input written with match: 0 contributes 0 when its signal matches; only
// an omitted match defaults to 1.
func TestProjectionInputExplicitZeroMatch(t *testing.T) {
	classifier := recipeClassifier(t, `version: v0.5
recipes:
  - name: default
    signals:
      keywords:
        - name: greeting
          operator: OR
          keywords: ["hello"]
    projections:
      scores:
        - name: substance
          method: weighted_sum
          inputs:
            - {type: keyword, name: greeting, weight: 1, match: 0, miss: 1}
        - name: greeted
          method: weighted_sum
          inputs:
            - {type: keyword, name: greeting, weight: 1}
      mappings:
        - name: substance_band
          source: substance
          outputs:
            - {name: substantive, gte: 0.5}
        - name: greeted_band
          source: greeted
          outputs:
            - {name: greeting_seen, gte: 0.5}
    decisions:
      - name: big
        priority: 100
        rules: {operator: OR, conditions: [{type: projection, name: substantive}, {type: projection, name: greeting_seen}]}
        modelRefs: [{model: local}]
      - name: fallback
        priority: 1
        rules: {operator: AND}
        modelRefs: [{model: local}]
providers:
  models:
    - name: local
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
`, nil)
	for _, tc := range []struct {
		text              string
		substance, greets float64
	}{
		{text: "hello there", substance: 0, greets: 1},
		{text: "explain quicksort", substance: 1, greets: 0},
	} {
		results := classifier.EvaluateAllSignalsWithForceOption(tc.text, false)
		if results.ProjectionScores["substance"] != tc.substance || results.ProjectionScores["greeted"] != tc.greets {
			t.Fatalf("text=%q keywords=%v scores=%v, want substance=%g greeted=%g", tc.text, results.MatchedKeywordRules, results.ProjectionScores, tc.substance, tc.greets)
		}
	}
}
