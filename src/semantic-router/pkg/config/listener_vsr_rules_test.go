package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const listenerRulesDocument = `version: v0.5
listeners:
  - name: agents
    port: 8899
    apis:
      chat_completions: {}
      systemone:
        vsr:
          rules: %s
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
          protocol: http
          provider: vllm
entrypoints:
  - {model_names: [vllm-sr/auto], recipe: default, default: true}
global:
  model_catalog:
    deployments:
      vela3: {provider: model_runtime, artifact: vllm-sr/vela3-lite, device: cpu}
recipes:
  - name: default
    signals:
      keywords:
        - {name: prod_database, operator: OR, method: regex, keywords: ['prod-db\.internal']}
        - {name: billing, operator: AND, keywords: [invoice, refund]}
        - {name: employee_id, operator: OR, method: regex, keywords: ['\bE-\d{6}\b']}
      systemone:
        - name: deletes_data
          deployment: vela3
          question:
            type: noul
            instructions: Does this command delete data that cannot be restored?
        - name: risk
          deployment: vela3
          question:
            type: choice
            instructions: What is the risk of this command?
            choices: [{key: safe}, {key: risky}]
  - name: unrouted
    signals:
      keywords:
        - {name: prod_database, operator: OR, keywords: [prod]}
`

func loadListenerRules(t *testing.T, rules string) (*RouterConfig, error) {
	t.Helper()
	return ParseYAMLBytes([]byte(strings.Replace(listenerRulesDocument, "%s", rules, 1)))
}

// A listener binds rules of a reachable recipe to guard and redact: guard
// takes a rule that says yes or no, redact one that finds text, and each name
// once.
func TestListenerRulesLoadOnlyWhatEachAppCanUse(t *testing.T) {
	cfg, err := loadListenerRules(t, `{recipe: default, guard: [{type: keyword, name: prod_database, enforce: required}, {type: keyword, name: billing}, {type: systemone, name: deletes_data}], redact: [{type: keyword, name: employee_id}]}`)
	if err != nil {
		t.Fatalf("rules each app can use were refused: %v", err)
	}
	rules := cfg.Listeners[0].APIs["systemone"].VSR.Rules
	if len(rules.Guard) != 3 || !rules.Guard[0].Required() || rules.Guard[1].Required() || rules.Redact[0].Name != "employee_id" {
		t.Fatalf("rules = %+v", rules)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := ParseYAMLBytes(exported)
	if err != nil || len(reloaded.Listeners[0].APIs["systemone"].VSR.Rules.Guard) != 3 {
		t.Fatalf("the rules did not survive export: %v\n%s", err, exported)
	}
	for rules, want := range map[string]string{
		`{recipe: missing, guard: [{type: keyword, name: prod_database}]}`:                                                 `"missing" names no recipe`,
		`{recipe: unrouted, guard: [{type: keyword, name: prod_database}]}`:                                                `recipe "unrouted" is not reached from an entrypoint`,
		`{recipe: default, guard: [{type: keyword, name: nothing}]}`:                                                       `"nothing" names no keyword rule`,
		`{recipe: default, guard: [{type: systemone, name: risk}]}`:                                                        "has no labels, so it never matches on its own",
		`{recipe: default, guard: [{type: pii, name: personal}]}`:                                                          `type "pii" is not a rule the plugins can ask`,
		`{recipe: default, redact: [{type: keyword, name: billing}]}`:                                                      "operator AND matches on which keywords occur",
		`{recipe: default, redact: [{type: systemone, name: deletes_data}]}`:                                               "only a span question finds text",
		`{recipe: default, guard: [{type: keyword, name: billing}, {type: keyword, name: billing}]}`:                       `"billing" is bound twice`,
		`{recipe: default, guard: [{type: keyword, name: billing, enforce: always}]}`:                                      `"always" is not required or advisory`,
		`{recipe: default, guard: [{type: keyword, name: billing}], redact: [{type: keyword, name: employee_id, why: x}]}`: `unknown field "why"`,
	} {
		if _, err := loadListenerRules(t, rules); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s\n got %v\nwant %q", rules, err, want)
		}
	}
	many := make([]string, MaxVSRRulesPerApp+1)
	for i := range many {
		many[i] = "{type: keyword, name: billing}"
	}
	if _, err := loadListenerRules(t, `{recipe: default, guard: [`+strings.Join(many, ", ")+`]}`); err == nil || !strings.Contains(err.Error(), "more than the 16") {
		t.Errorf("17 guard rules loaded: %v", err)
	}
}
