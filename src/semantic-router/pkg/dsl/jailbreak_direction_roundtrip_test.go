package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A jailbreak rule's direction has to survive both directions of the DSL, or
// a config that scores the model's output comes back as one that scores the
// prompt after a round trip through the CLI or the dashboard.
const jailbreakDirectionDSL = `
SIGNAL jailbreak prompt_injection { threshold: 0.8 }
SIGNAL jailbreak unsafe_completion { direction: "response" threshold: 0.85 description: "Scores the model's output." }
ROUTE guarded { PRIORITY 1 WHEN jailbreak("prompt_injection") MODEL "m:1b" }
`

func compileJailbreakDirection(t *testing.T, input string) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.JailbreakRules) != 2 {
		t.Fatalf("expected 2 jailbreak rules, got %d", len(cfg.JailbreakRules))
	}
	return cfg
}

func TestJailbreakDirectionCompiles(t *testing.T) {
	cfg := compileJailbreakDirection(t, jailbreakDirectionDSL)
	if got := cfg.JailbreakRules[0].Direction; got != "" {
		t.Errorf("prompt_injection direction = %q, want empty (request by default)", got)
	}
	if got := cfg.JailbreakRules[1].Direction; got != config.SignalDirectionResponse {
		t.Errorf("unsafe_completion direction = %q, want %q", got, config.SignalDirectionResponse)
	}
}

func TestJailbreakDirectionDecompileRoundTrip(t *testing.T) {
	cfg := compileJailbreakDirection(t, jailbreakDirectionDSL)

	dslText, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	if !strings.Contains(dslText, `direction: "response"`) {
		t.Errorf("decompiled DSL dropped the direction:\n%s", dslText)
	}
	if strings.Count(dslText, "direction:") != 1 {
		t.Errorf("a request-direction rule must not gain an explicit direction:\n%s", dslText)
	}

	again := compileJailbreakDirection(t, dslText)
	if got := again.JailbreakRules[1].Direction; got != config.SignalDirectionResponse {
		t.Errorf("direction after round trip = %q", got)
	}
}

func TestJailbreakDirectionASTDecompile(t *testing.T) {
	cfg := compileJailbreakDirection(t, jailbreakDirectionDSL)

	prog := DecompileToAST(cfg)
	var found bool
	for _, sig := range prog.Signals {
		if sig.SignalType == "jailbreak" && sig.Name == "unsafe_completion" {
			v, ok := sig.Fields["direction"].(StringValue)
			found = ok && v.V == config.SignalDirectionResponse
		}
	}
	if !found {
		t.Error("AST decompile dropped the direction field")
	}
}
