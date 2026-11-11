package classification

import (
	"slices"
	"testing"
)

// A projection that reads an authz role sees the role the request's
// identity holds, as a decision that reads it directly does.
func TestAuthzRoleReachesProjectionScores(t *testing.T) {
	classifier := recipeClassifier(t, `version: v0.5
recipes:
  - name: default
    signals:
      role_bindings:
        - name: premium-binding
          role: premium_user
          subjects:
            - kind: Group
              name: premium-tier
    projections:
      scores:
        - name: priority_score
          method: weighted_sum
          inputs:
            - {type: authz, name: premium_user, weight: 1}
      mappings:
        - name: priority_band
          source: priority_score
          outputs:
            - {name: priority_high, gte: 0.5}
    decisions:
      - name: premium_by_projection
        priority: 200
        rules: {operator: AND, conditions: [{type: projection, name: priority_high}]}
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
	results, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
		Text: "hi", ContextText: "hi", CurrentUserText: "hi",
		Headers: map[string]string{classifier.authzUserIDHeader: "alice", classifier.authzUserGroupsHeader: "premium-tier"},
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := classifier.EvaluateDecisionWithEngine(results)
	if err != nil || decision == nil {
		t.Fatalf("decision=%v err=%v", decision, err)
	}
	if results.ProjectionScores["priority_score"] != 1 || !slices.Contains(results.MatchedProjectionRules, "priority_high") || decision.Decision.Name != "premium_by_projection" {
		t.Fatalf("authz=%v scores=%v projections=%v decision=%q, want the premium role to score 1 and route by projection",
			results.MatchedAuthzRules, results.ProjectionScores, results.MatchedProjectionRules, decision.Decision.Name)
	}
}
