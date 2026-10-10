package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A hallucination rule has to survive both directions of the DSL, or a config
// that checks the model's answer comes back without the check after a round
// trip through the CLI or the dashboard.
const hallucinationSignalDSL = `
SIGNAL keyword probe { keywords: ["__probe__"] }
SIGNAL hallucination ungrounded_claims { description: "Checks the answer against its grounding context." }
SIGNAL hallucination plain_check {}
ROUTE grounded { PRIORITY 1 WHEN keyword("probe") MODEL "m:1b" }
`

func compileHallucinationSignal(t *testing.T, input string) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.HallucinationRules) != 2 {
		t.Fatalf("expected 2 hallucination rules, got %d", len(cfg.HallucinationRules))
	}
	return cfg
}

func TestHallucinationSignalCompiles(t *testing.T) {
	cfg := compileHallucinationSignal(t, hallucinationSignalDSL)
	rule := cfg.HallucinationRules[0]
	if rule.Name != "ungrounded_claims" || rule.Description != "Checks the answer against its grounding context." {
		t.Errorf("first rule = %+v", rule)
	}
	if plain := cfg.HallucinationRules[1]; plain.Name != "plain_check" {
		t.Errorf("second rule = %+v, want plain_check", plain)
	}
}

func TestHallucinationSignalDecompileRoundTrip(t *testing.T) {
	cfg := compileHallucinationSignal(t, hallucinationSignalDSL)

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	if !strings.Contains(dslText, "SIGNAL hallucination ungrounded_claims") || !strings.Contains(dslText, "SIGNAL hallucination plain_check") {
		t.Errorf("decompiled DSL dropped a hallucination rule:\n%s", dslText)
	}
	if again := compileHallucinationSignal(t, dslText); again.HallucinationRules[0].Description != cfg.HallucinationRules[0].Description {
		t.Errorf("description after round trip = %q", again.HallucinationRules[0].Description)
	}
}

func TestHallucinationSignalRejectsRetiredNLI(t *testing.T) {
	_, errs := Compile(`
SIGNAL keyword probe { keywords: ["__probe__"] }
SIGNAL hallucination ungrounded_claims { use_nli: true }
ROUTE grounded { PRIORITY 1 WHEN keyword("probe") MODEL "m:1b" }
`)
	if len(errs) == 0 || !strings.Contains(errs[0].Error(), "use_nli is retired") {
		t.Fatalf("use_nli must be refused, got %v", errs)
	}
}

func TestHallucinationSignalASTDecompile(t *testing.T) {
	cfg := compileHallucinationSignal(t, hallucinationSignalDSL)

	prog := DecompileToAST(cfg)
	var found bool
	for _, sig := range prog.Signals {
		if sig.SignalType == "hallucination" && sig.Name == "ungrounded_claims" {
			v, ok := sig.Fields["description"].(StringValue)
			found = ok && v.V != ""
		}
	}
	if !found {
		t.Error("AST decompile dropped the hallucination rule or its description")
	}
}
