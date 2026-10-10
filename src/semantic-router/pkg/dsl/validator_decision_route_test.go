package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const taskQuestionDSL = `
SIGNAL decision task {
  question: { type: "choice", instructions: "What kind of work?", choices: [{ key: "agentic" }, { key: "facts" }] }
}
`

func guardWarnings(t *testing.T, input string) []string {
	t.Helper()
	diags, errs := Validate(input)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	var warnings []string
	for _, d := range diags {
		if strings.Contains(d.Message, "no mutual exclusion guard") {
			warnings = append(warnings, d.Message)
		}
	}
	return warnings
}

func TestRoutesOnDifferentOptionsOfOneQuestionDoNotOverlap(t *testing.T) {
	distinct := taskQuestionDSL + `
ROUTE agentic { PRIORITY 200 WHEN decision("task", label: "agentic") MODEL "m1" }
ROUTE facts { PRIORITY 100 WHEN decision("task", label: "facts") MODEL "m2" }
`
	if warnings := guardWarnings(t, distinct); len(warnings) > 0 {
		t.Fatalf("different options of one question are different references: %v", warnings)
	}
	same := taskQuestionDSL + `
ROUTE agentic { PRIORITY 200 WHEN decision("task", label: "agentic") MODEL "m1" }
ROUTE tools { PRIORITY 100 WHEN decision("task", label: "agentic") MODEL "m2" }
`
	warnings := guardWarnings(t, same)
	if len(warnings) != 1 || !strings.Contains(warnings[0], `decision("task:agentic")`) {
		t.Fatalf("the same option in two routes must still warn, got %v", warnings)
	}
}

func TestAFastResponseRouteNeedsNoModel(t *testing.T) {
	for name, input := range map[string]string{
		"inline": `
SIGNAL jailbreak attack { threshold: 0.9 }
ROUTE guard { PRIORITY 100 WHEN jailbreak("attack") PLUGIN fast_response { message: "Declined." } }
`,
		"template": `
SIGNAL jailbreak attack { threshold: 0.9 }
PLUGIN decline fast_response { message: "Declined." }
ROUTE guard { PRIORITY 100 WHEN jailbreak("attack") PLUGIN decline }
`,
	} {
		diags, errs := Validate(input)
		if len(errs) > 0 {
			t.Fatalf("%s: %v", name, errs)
		}
		for _, d := range diags {
			if strings.Contains(d.Message, "has no MODEL specified") {
				t.Fatalf("%s: a fast_response route calls no model: %s", name, d.Message)
			}
		}
	}
	diags, _ := Validate(`ROUTE plain { PRIORITY 100 WHEN keyword("x") }`)
	found := false
	for _, d := range diags {
		found = found || strings.Contains(d.Message, "has no MODEL specified")
	}
	if !found {
		t.Fatal("a route without a model or a fast_response still needs a MODEL")
	}
}

func TestDecompileLeavesACatalogModelsSizeToItsCard(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{
		"catalog-model": {Catalog: "vendor/catalog-model", ParamSize: "27B"},
		"custom-model":  {ParamSize: "7B"},
	}
	cfg.Decisions = []config.Decision{{
		Name: "route", Priority: 10,
		Rules:     config.RuleCombination{Type: "keyword", Name: "x"},
		ModelRefs: []config.ModelRef{{Model: "catalog-model"}, {Model: "custom-model"}},
	}}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(source, `param_size = "27B"`) || !strings.Contains(source, `param_size = "7B"`) {
		t.Fatalf("a route repeats only an operator's own model size:\n%s", source)
	}
}
