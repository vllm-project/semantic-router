package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFilterDecisionsByAuthz(t *testing.T) {
	authz := func(name string) config.RuleNode {
		return config.RuleNode{Type: config.SignalTypeAuthz, Name: name}
	}
	keyword := func(name string) config.RuleNode {
		return config.RuleNode{Type: config.SignalTypeKeyword, Name: name}
	}
	decision := func(name string, rules config.RuleNode) config.Decision {
		return config.Decision{Name: name, Rules: rules}
	}

	tests := []struct {
		name         string
		decisions    []config.Decision
		matchedRoles []string
		wantNames    []string
	}{
		{
			name: "drops an AND branch denied by authz",
			decisions: []config.Decision{
				decision("route-a", config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{authz("route-a"), keyword("math")}}),
				decision("route-b", config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{authz("route-b"), keyword("code")}}),
			},
			matchedRoles: []string{"route-a"},
			wantNames:    []string{"route-a"},
		},
		{
			name: "keeps an OR branch when a non-authz child is unknown",
			decisions: []config.Decision{
				decision("fallback", config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{authz("route-b"), keyword("general")}}),
			},
			matchedRoles: []string{"route-a"},
			wantNames:    []string{"fallback"},
		},
		{
			name: "applies unary NOT to known authz facts",
			decisions: []config.Decision{
				decision("not-route-b", config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{authz("route-b")}}),
				decision("not-route-a", config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{authz("route-a")}}),
			},
			matchedRoles: []string{"route-a"},
			wantNames:    []string{"not-route-b"},
		},
		{
			name: "preserves conservative candidates for unknown and malformed rules",
			decisions: []config.Decision{
				decision("non-authz", keyword("general")),
				decision("malformed-not", config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{authz("route-a"), authz("route-b")}}),
				decision("unsupported", config.RuleNode{Operator: "XOR", Conditions: []config.RuleNode{authz("route-b")}}),
			},
			matchedRoles: []string{"route-a"},
			wantNames:    []string{"non-authz", "malformed-not", "unsupported"},
		},
		{
			name: "matches the decision engine identities for empty AND and OR",
			decisions: []config.Decision{
				decision("empty-and", config.RuleNode{Operator: "AND"}),
				decision("empty-or", config.RuleNode{Operator: "OR"}),
			},
			wantNames: []string{"empty-and"},
		},
		{
			name: "matches role names case-sensitively",
			decisions: []config.Decision{
				decision("exact", authz("Premium")),
				decision("different-case", authz("premium")),
			},
			matchedRoles: []string{"Premium"},
			wantNames:    []string{"exact"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterDecisionsByAuthz(tt.decisions, tt.matchedRoles)
			gotNames := make([]string, 0, len(got))
			for _, candidate := range got {
				gotNames = append(gotNames, candidate.Name)
			}
			if !reflect.DeepEqual(gotNames, tt.wantNames) {
				t.Fatalf("filterDecisionsByAuthz() names = %v, want %v", gotNames, tt.wantNames)
			}
		})
	}
}

func TestGetUsedSignalsForDecisions(t *testing.T) {
	routeA := config.Decision{
		Name: "route-a",
		Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
			{Type: config.SignalTypeAuthz, Name: "route-a"},
			{Type: config.SignalTypeKeyword, Name: "math"},
		}},
	}
	routeB := config.Decision{
		Name: "route-b",
		Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
			{Type: config.SignalTypeAuthz, Name: "route-b"},
			{Type: config.SignalTypeJailbreak, Name: "guard"},
		}},
	}
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{routeA, routeB}},
	}}

	used := classifier.getUsedSignalsForDecisions([]config.Decision{routeA})
	if !used[config.SignalTypeKeyword+":math"] {
		t.Fatalf("expected route-a keyword signal to be used, got %v", used)
	}
	if used[config.SignalTypeJailbreak+":guard"] {
		t.Fatalf("did not expect route-b jailbreak signal to be used, got %v", used)
	}
}

type authzScopeTokenCounter struct {
	calls int
}

func (c *authzScopeTokenCounter) CountTokens(string) (int, error) {
	c.calls++
	return 1, nil
}

func TestEvaluateAllSignalsWithHeadersForDecisionsScopesNonAuthzWork(t *testing.T) {
	decisions := []config.Decision{
		{
			Name: "route-a",
			Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
				{Type: config.SignalTypeAuthz, Name: "route-a"},
			}},
		},
		{
			Name: "route-b",
			Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
				{Type: config.SignalTypeAuthz, Name: "route-b"},
				{Type: config.SignalTypeContext, Name: "short"},
			}},
		},
	}
	bindings := []config.RoleBinding{
		{Name: "route-a-users", Role: "route-a", Subjects: []config.Subject{{Kind: "User", Name: "alice"}}},
		{Name: "route-b-users", Role: "route-b", Subjects: []config.Subject{{Kind: "User", Name: "bob"}}},
	}
	authzClassifier, err := NewAuthzClassifier(bindings)
	if err != nil {
		t.Fatalf("NewAuthzClassifier() error = %v", err)
	}
	counter := &authzScopeTokenCounter{}
	contextRules := []config.ContextRule{{Name: "short", MinTokens: "0", MaxTokens: "10"}}
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals:   config.Signals{RoleBindings: bindings, ContextRules: contextRules},
			Decisions: decisions,
		}},
		authzClassifier:       authzClassifier,
		authzUserIDHeader:     "x-authz-user-id",
		authzUserGroupsHeader: "x-authz-user-groups",
		contextClassifier:     NewContextClassifier(counter, contextRules),
	}

	results, candidates, err := classifier.EvaluateAllSignalsWithHeadersForDecisions(
		"hello", "hello", "hello", nil, nil, false,
		map[string]string{"x-authz-user-id": "alice"}, false, "", decisions,
	)
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeadersForDecisions() error = %v", err)
	}
	if !reflect.DeepEqual(results.MatchedAuthzRules, []string{"route-a"}) {
		t.Fatalf("matched authz rules = %v, want [route-a]", results.MatchedAuthzRules)
	}
	if got := decisionNames(candidates); !reflect.DeepEqual(got, []string{"route-a"}) {
		t.Fatalf("candidate decisions = %v, want [route-a]", got)
	}
	if counter.calls != 0 {
		t.Fatalf("irrelevant context classifier calls = %d, want 0", counter.calls)
	}

	results, candidates, err = classifier.EvaluateAllSignalsWithHeadersForDecisions(
		"hello", "hello", "hello", nil, nil, false,
		map[string]string{"x-authz-user-id": "alice"}, true, "", decisions,
	)
	if err != nil {
		t.Fatalf("force-all EvaluateAllSignalsWithHeadersForDecisions() error = %v", err)
	}
	if got := decisionNames(candidates); !reflect.DeepEqual(got, []string{"route-a", "route-b"}) {
		t.Fatalf("force-all candidate decisions = %v, want all decisions", got)
	}
	if counter.calls != 1 {
		t.Fatalf("force-all context classifier calls = %d, want 1", counter.calls)
	}
	if !reflect.DeepEqual(results.MatchedContextRules, []string{"short"}) {
		t.Fatalf("force-all context rules = %v, want [short]", results.MatchedContextRules)
	}
}

func decisionNames(decisions []config.Decision) []string {
	names := make([]string, 0, len(decisions))
	for _, decision := range decisions {
		names = append(names, decision.Name)
	}
	return names
}
