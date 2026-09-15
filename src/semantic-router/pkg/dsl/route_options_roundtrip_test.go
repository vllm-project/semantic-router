package dsl

import "testing"

func TestDecompileRouteOptionsRecompile(t *testing.T) {
	cfg, errs := Compile(`
SIGNAL domain math { description: "math" }

ROUTE math_route (description = "Math route" on_unknown = "no_match") {
  PRIORITY 100
  WHEN domain("math")
  MODEL "math-model"
}`)
	if len(errs) > 0 {
		t.Fatalf("Compile returned errors: %v", errs)
	}

	decompiled, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile returned an error: %v", err)
	}
	recompiled, errs := Compile(decompiled)
	if len(errs) > 0 {
		t.Fatalf("decompiled DSL failed to recompile: %v\n%s", errs, decompiled)
	}

	decision := recompiled.Decisions[0]
	if decision.Description != "Math route" || decision.Rules.OnUnknown != "no_match" {
		t.Fatalf("route options were not preserved: %#v", decision)
	}
}
