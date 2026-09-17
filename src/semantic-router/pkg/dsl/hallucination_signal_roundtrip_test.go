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
SIGNAL hallucination ungrounded_claims { use_nli: true description: "Checks the answer against its grounding context." }
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
	if rule.Name != "ungrounded_claims" || !rule.UseNLI || rule.Description != "Checks the answer against its grounding context." {
		t.Errorf("first rule = %+v", rule)
	}
	if plain := cfg.HallucinationRules[1]; plain.Name != "plain_check" || plain.UseNLI {
		t.Errorf("second rule = %+v, want plain_check without NLI", plain)
	}
}

func TestHallucinationSignalDecompileRoundTrip(t *testing.T) {
	cfg := compileHallucinationSignal(t, hallucinationSignalDSL)

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	if !strings.Contains(dslText, "SIGNAL hallucination ungrounded_claims") || !strings.Contains(dslText, "use_nli: true") {
		t.Errorf("decompiled DSL dropped the hallucination rule or its use_nli:\n%s", dslText)
	}
	if strings.Count(dslText, "use_nli:") != 1 {
		t.Errorf("a rule without NLI must not gain an explicit use_nli:\n%s", dslText)
	}

	again := compileHallucinationSignal(t, dslText)
	if !again.HallucinationRules[0].UseNLI || again.HallucinationRules[1].UseNLI {
		t.Errorf("use_nli after round trip = %v / %v", again.HallucinationRules[0].UseNLI, again.HallucinationRules[1].UseNLI)
	}
}

func TestHallucinationSignalASTDecompile(t *testing.T) {
	cfg := compileHallucinationSignal(t, hallucinationSignalDSL)

	prog := DecompileToAST(cfg)
	var found bool
	for _, sig := range prog.Signals {
		if sig.SignalType == "hallucination" && sig.Name == "ungrounded_claims" {
			v, ok := sig.Fields["use_nli"].(BoolValue)
			found = ok && v.V
		}
	}
	if !found {
		t.Error("AST decompile dropped the hallucination rule or its use_nli field")
	}
}
