package config

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestParseYAMLBytesRejectsOversizedDecisionBeforeStaticValidation(t *testing.T) {
	rule := map[string]interface{}{"type": "keyword", "name": "urgent"}
	for depth := 1; depth < 17; depth++ {
		rule = map[string]interface{}{"operator": "AND", "conditions": []interface{}{rule}}
	}
	document := map[string]interface{}{
		"version": "v0.3",
		"global":  map[string]interface{}{"router": map[string]interface{}{"config_source": "kubernetes"}},
		"routing": map[string]interface{}{"decisions": []interface{}{
			map[string]interface{}{"name": "oversized", "rules": rule},
		}},
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ParseYAMLBytes(data)
	if err == nil || !strings.Contains(err.Error(), `decision "oversized"`) || !strings.Contains(err.Error(), "depth 17 exceeds max_depth=16") {
		t.Fatalf("expected a named decision depth-limit error, got %v", err)
	}
}

type decisionRuleCorpusCase struct {
	Name   string          `json:"name"`
	Kind   string          `json:"kind"`
	Size   int             `json:"size"`
	Limits json.RawMessage `json:"limits"`
	Error  string          `json:"error"`
}

func ruleCorpusDocument(t *testing.T, tc decisionRuleCorpusCase) map[string]interface{} {
	t.Helper()
	rule := map[string]interface{}{"type": "keyword", "name": "urgent"}
	switch tc.Kind {
	case "depth":
		for depth := 1; depth < tc.Size; depth++ {
			rule = map[string]interface{}{"operator": "AND", "conditions": []interface{}{rule}}
		}
	case "nodes":
		children := make([]interface{}, tc.Size-1)
		for i := range children {
			children[i] = rule
		}
		rule = map[string]interface{}{"operator": "AND", "conditions": children}
	default:
		rule = map[string]interface{}{}
	}
	decision := map[string]interface{}{"name": "bounded", "rules": rule}
	if tc.Kind == "missing" {
		delete(decision, "rules")
	}
	router := map[string]interface{}{"config_source": "kubernetes"}
	if len(tc.Limits) > 0 {
		var limits interface{}
		if err := yaml.Unmarshal(tc.Limits, &limits); err != nil {
			t.Fatal(err)
		}
		router["decision_rule_limits"] = limits
	}
	return map[string]interface{}{
		"version": "v0.3",
		"global":  map[string]interface{}{"router": router},
		"routing": map[string]interface{}{"decisions": []interface{}{decision}},
	}
}

func TestDecisionRuleLimitsSharedCorpus(t *testing.T) {
	data, err := os.ReadFile("testdata/decision_rule_limits.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []decisionRuleCorpusCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			document := ruleCorpusDocument(t, tc)
			data, err := yaml.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			parsers := map[string]func([]byte) (*RouterConfig, error){
				"runtime": ParseYAMLBytes, "offline": ParseYAMLBytesWithoutEnvExpansion,
				"editor": ParseYAMLBytesDeferringEnv,
			}
			for name, parse := range parsers {
				t.Run(name, func(t *testing.T) {
					cfg, err := parse(data)
					if tc.Error != "" {
						if err == nil || !strings.Contains(err.Error(), tc.Error) {
							t.Fatalf("expected %q, got %v", tc.Error, err)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					if validationErr := ValidateDecisionRuleLimits(cfg); validationErr != nil {
						t.Fatalf("typed validation differs: %v", validationErr)
					}
					global := CanonicalGlobalFromRouterConfig(cfg)
					depth, nodes, err := global.Router.DecisionRuleLimits.Effective()
					if err != nil || depth < 1 || nodes < 1 {
						t.Fatalf("export lost limits: %v", global.Router.DecisionRuleLimits)
					}
				})
			}
		})
	}
}

func TestDecisionRuleLimitsTypedAndRecipeBudgets(t *testing.T) {
	leaf := RuleNode{Type: "keyword", Name: "urgent"}
	cfg := &RouterConfig{
		DecisionRuleLimits: DecisionRuleLimits{MaxDepth: canonicalIntPtr(1), MaxNodes: canonicalIntPtr(1)},
		IntelligentRouting: IntelligentRouting{Decisions: []Decision{{Name: "a", Rules: leaf}, {Name: "b", Rules: leaf}}},
		Recipes:            []RoutingRecipe{{Name: "support", Profile: RoutingProfile{Decisions: []Decision{{Name: "c", Rules: leaf}}}}},
	}
	if err := ValidateDecisionRuleLimits(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Recipes[0].Profile.Decisions[0].Rules = RuleNode{Operator: "AND", Conditions: []RuleNode{leaf}}
	err := ValidateKubernetesConfigContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), `routing recipe "support": decision "c": rules.conditions[0]`) {
		t.Fatalf("expected recipe-scoped early error, got %v", err)
	}
	cfg.DecisionRuleLimits.MaxDepth = canonicalIntPtr(0)
	if err := ValidateDecisionRuleLimits(cfg); err == nil {
		t.Fatal("expected explicit typed zero to fail")
	}
}

func TestDecisionRuleLimitsRawRecipesAndEarlyRejection(t *testing.T) {
	document := ruleCorpusDocument(t, decisionRuleCorpusCase{Kind: "depth", Size: 1000})
	document["recipes"] = []interface{}{map[string]interface{}{"name": "support", "routing": document["routing"]}}
	delete(document, "routing")
	_, err := validateRawDecisionRuleLimits(document)
	if err == nil || !strings.Contains(err.Error(), `routing recipe "support": decision "bounded"`) || !strings.Contains(err.Error(), "depth 17") {
		t.Fatalf("expected early recipe depth rejection, got %v", err)
	}
}
