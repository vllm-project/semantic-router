package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const routeActionDSL = `
MODEL "safe-model" {}

SIGNAL jailbreak "prompt_injection" {
  type: "prompt_guard"
}

ROUTE "prompt_attack_guard_route" {
  PRIORITY 120
  WHEN jailbreak("prompt_injection")
  ACTION route "safe-model"
  MODEL "safe-model"
}`

func compileRouteActionDSL(t *testing.T, source string) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(source)
	if len(errs) != 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	return cfg
}

func TestCompilePreservesRouteAction(t *testing.T) {
	cfg := compileRouteActionDSL(t, routeActionDSL)
	if len(cfg.Decisions) != 1 {
		t.Fatalf("decisions = %d, want 1", len(cfg.Decisions))
	}
	action := cfg.Decisions[0].Action
	if action == nil || action.Type != config.DecisionActionRoute || action.Destination != "safe-model" {
		t.Fatalf("action = %+v, want route to safe-model", action)
	}
}

func TestDecompileRoundTripPreservesRouteAction(t *testing.T) {
	cfg := compileRouteActionDSL(t, routeActionDSL)

	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	if !strings.Contains(source, `ACTION route "safe-model"`) {
		t.Fatalf("decompiled source drops the action:\n%s", source)
	}

	roundTrip := compileRouteActionDSL(t, source)
	action := roundTrip.Decisions[0].Action
	if action == nil || action.Type != config.DecisionActionRoute || action.Destination != "safe-model" {
		t.Fatalf("round-trip action = %+v, want route to safe-model", action)
	}
}

func TestValidateRouteActionConstraints(t *testing.T) {
	tests := []struct {
		name     string
		source   string
		wantDiag string
	}{
		{
			"valid",
			routeActionDSL,
			"",
		},
		{
			"missing jailbreak condition",
			`
MODEL "safe-model" {}

SIGNAL keyword "hack" {
  operator: "contains"
  values: ["hack"]
}

ROUTE "guard" {
  PRIORITY 120
  WHEN keyword("hack")
  ACTION route "safe-model"
  MODEL "safe-model"
}`,
			"requires an explicit jailbreak condition",
		},
		{
			"unknown action type",
			`
MODEL "safe-model" {}

SIGNAL jailbreak "prompt_injection" {
  type: "prompt_guard"
}

ROUTE "guard" {
  PRIORITY 120
  WHEN jailbreak("prompt_injection")
  ACTION block "safe-model"
  MODEL "safe-model"
}`,
			"action type must be",
		},
		{
			"undeclared destination",
			`
MODEL "other-model" {}

SIGNAL jailbreak "prompt_injection" {
  type: "prompt_guard"
}

ROUTE "guard" {
  PRIORITY 120
  WHEN jailbreak("prompt_injection")
  ACTION route "safe-model"
  MODEL "other-model"
}`,
			"is not a declared model",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			diagnostics, errs := Validate(test.source)
			if len(errs) != 0 {
				t.Fatalf("parse errors: %v", errs)
			}
			if test.wantDiag == "" {
				if message := diagnosticContaining(diagnostics, "action"); message != "" {
					t.Fatalf("unexpected action diagnostic: %s", message)
				}
				return
			}
			if diagnosticContaining(diagnostics, test.wantDiag) == "" {
				t.Fatalf("no diagnostic contains %q: %#v", test.wantDiag, diagnostics)
			}
		})
	}
}

func diagnosticContaining(diagnostics []Diagnostic, substring string) string {
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, substring) {
			return diagnostic.Message
		}
	}
	return ""
}
