package classification

import (
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// recipeClassifier loads a config document the way the router does and
// returns its default recipe's classifier, asking decider when one is given.
func recipeClassifier(t *testing.T, document string, decider modelservice.Decider) *Classifier {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	classifier := classifiers.Default()
	if decider != nil {
		classifier.SetDecisionDecider(decider)
	}
	return classifier
}

// partitionConfig has a partition whose default is the only member a
// decision reads.
const partitionConfig = `version: v0.5
recipes:
  - name: default
    signals:
      domains:
        - name: business
        - name: law
        - name: health
    projections:
      partitions:
        - name: practice_areas
          semantics: exclusive
          members: [business, law, health]
          default: business
    decisions:
      - name: business_route
        priority: 200
        rules:
          operator: AND
          on_unknown: fail_request
          conditions: [{type: domain, name: business}]
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
`

// A partition picks its winner among all its members, so a decision that
// reads one member makes the others count too: a legal question is not
// business because no decision reads law, and routing agrees with a
// preview that evaluates every signal. A subject outside the partition
// still gets the default.
func TestPartitionEvaluatesEveryMemberWhenADecisionReadsOne(t *testing.T) {
	for _, tc := range []struct{ choice, want string }{
		{choice: "law", want: "fallback"},
		{choice: "math", want: "business_route"},
	} {
		decider := newFakeDecider(map[string]modelservice.Answer{
			presetQuestionID(presets.Domain): {Type: "choice", Choice: tc.choice, Probabilities: map[string]float64{tc.choice: 0.9}},
		})
		classifier := recipeClassifier(t, partitionConfig, decider)
		for _, force := range []bool{false, true} {
			results := classifier.EvaluateAllSignalsWithForceOption("Is a verbal contract binding under common law?", force)
			decision, err := classifier.EvaluateDecisionWithEngine(results)
			if err != nil || decision == nil {
				t.Fatalf("choice=%s force=%v: decision=%v err=%v", tc.choice, force, decision, err)
			}
			if decision.Decision.Name != tc.want {
				t.Fatalf("choice=%s force=%v: domains=%v decision=%q, want %q", tc.choice, force, results.MatchedDomainRules, decision.Decision.Name, tc.want)
			}
		}
	}
}

// A partition's default means no member matched, which a failed member
// cannot tell: the failure reaches the decision, whose on_unknown policy
// decides, instead of becoming a match of the default.
func TestPartitionDoesNotDefaultOverAFailedMember(t *testing.T) {
	classifier := recipeClassifier(t, partitionConfig, newFakeDecider(nil))
	results := classifier.EvaluateAllSignalsWithForceOption("What are the elements of a valid contract?", false)
	if _, failed := results.SignalErrors["domain:business"]; !failed {
		t.Fatalf("errors=%v, want the domain question to fail", results.SignalErrors)
	}
	decision, err := classifier.EvaluateDecisionWithEngine(results)
	if err == nil || slices.Contains(results.MatchedDomainRules, "business") {
		t.Fatalf("domains=%v decision=%v err=%v, want business unknown and on_unknown: fail_request to fail the request", results.MatchedDomainRules, decision, err)
	}
}

// splitPartitionConfig asks law on its own deployment, so law can fail while
// the other members are answered. DECISION is the decision that reads the
// default member.
const splitPartitionConfig = `version: v0.5
global:
  model_catalog:
    deployments:
      kai:
        provider: model_runtime
        endpoint: http://kai.invalid:8100
recipes:
  - name: default
    signals:
      domains:
        - name: business
        - name: law
          deployment: kai
        - name: health
    projections:
      partitions:
        - name: practice_areas
          semantics: exclusive
          members: [business, law, health]
          default: business
    decisions:
DECISION      - name: fallback
        priority: 1
        rules: {operator: AND}
        modelRefs: [{model: local}]
providers:
  models:
    - name: local
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
`

// When one member fails while the others are answered and none matched, the
// default is as unknown as the failed member: its decisions resolve through
// their on_unknown policy, whichever it is.
func TestPartitionDefaultIsUnknownWhenOneMemberFails(t *testing.T) {
	for _, tc := range []struct {
		name, decision, want string
		fails                bool
	}{
		{
			name:     "omitted",
			decision: "      - name: not_business\n        priority: 200\n        rules: {operator: NOT, conditions: [{type: domain, name: business}]}\n        modelRefs: [{model: local}]\n",
			want:     "fallback",
		},
		{
			name:     "match",
			decision: "      - name: business_route\n        priority: 200\n        rules: {operator: AND, on_unknown: match, conditions: [{type: domain, name: business}]}\n        modelRefs: [{model: local}]\n",
			want:     "business_route",
		},
		{
			name:     "fail_request",
			decision: "      - name: business_route\n        priority: 200\n        rules: {operator: AND, on_unknown: fail_request, conditions: [{type: domain, name: business}]}\n        modelRefs: [{model: local}]\n",
			fails:    true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			decider := newFakeDecider(map[string]modelservice.Answer{
				presetQuestionID(presets.Domain): {Type: "choice", Choice: "math", Probabilities: map[string]float64{"math": 0.9}},
			})
			decider.failures["kai"] = modelservice.ErrUnavailable
			classifier := recipeClassifier(t, strings.Replace(splitPartitionConfig, "DECISION", tc.decision, 1), decider)
			results := classifier.EvaluateAllSignalsWithForceOption("Is a verbal contract binding under common law?", false)
			if _, failed := results.SignalErrors["domain:law"]; !failed {
				t.Fatalf("errors=%v, want law to fail on its own deployment", results.SignalErrors)
			}
			decision, err := classifier.EvaluateDecisionWithEngine(results)
			name := ""
			if decision != nil {
				name = decision.Decision.Name
			}
			if tc.fails {
				if err == nil {
					t.Fatalf("errors=%v decision=%q, want fail_request to fail the request", results.SignalErrors, name)
				}
				return
			}
			if err != nil || name != tc.want {
				t.Fatalf("errors=%v decision=%q err=%v, want %q", results.SignalErrors, name, err, tc.want)
			}
		})
	}
}
