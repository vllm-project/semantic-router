package dsl

import (
	"fmt"
	"strings"
	"testing"
)

const guardedContextDSL = `
SIGNAL domain test { description: "test" }
SIGNAL hallucination ungrounded { description: "Claims the context does not support." }
ROUTE guarded {
  PRIORITY 1
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN response_guard {
    rules: [{ when: { type: "hallucination", name: "ungrounded", predicate: { gte: 0.9 } }, action: "block" }]
    on_unknown: "block"
  }
  PLUGIN context {
    enabled: true
    sources: [{ type: "memory", match: { judge: { question: { type: "choice", instructions: "Which memory answers the request?" } } } }]
  }
}
`

// Plugin payloads reach the emitted YAML under their YAML keys, nested types
// included, and without the fields they leave unset.
func TestPluginPayloadsKeepTheirYAMLKeysInEmittedYAML(t *testing.T) {
	emitted, err := EmitRoutingYAMLFromConfig(mustCompile(t, guardedContextDSL))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"gte: 0.9", "instructions: Which memory answers the request?"} {
		if !strings.Contains(string(emitted), want) {
			t.Errorf("emitted YAML lacks %q:\n%s", want, emitted)
		}
	}
	for _, wrong := range []string{"GTE", "LT:", "Instructions", "Choices"} {
		if strings.Contains(string(emitted), wrong) {
			t.Errorf("emitted YAML carries %q:\n%s", wrong, emitted)
		}
	}
}

// A field the Router would refuse at load fails the compile, so a typo cannot
// drop a fail-closed setting such as the guard's on_unknown.
func TestPluginFieldTyposFailToCompile(t *testing.T) {
	for _, typo := range []struct{ field, misspelled, key string }{
		{`on_unknown: "block"`, `onunknown: "block"`, "onunknown"},
		{`enabled: true`, `enable: true`, "enable"},
		{`PLUGIN context {`, `PLUGIN masking { enabled: true, signal: "contact", entity_type: ["EMAIL"] }
  PLUGIN context {`, "entity_type"},
	} {
		source := strings.Replace(guardedContextDSL, typo.field, typo.misspelled, 1)
		if _, errs := Compile(source); !strings.Contains(fmt.Sprint(errs), `unknown field "`+typo.key+`"`) {
			t.Errorf("%s compiled with errors %v", typo.misspelled, errs)
		}
	}
}
