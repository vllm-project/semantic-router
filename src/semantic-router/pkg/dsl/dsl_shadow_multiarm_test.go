package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompileShadowDispatchMultiArm(t *testing.T) {
	input := `SIGNAL domain math { description: "math" }

ROUTE test {
  PRIORITY 1
  WHEN domain("math")
  MODEL "m:1b"
  PLUGIN shadow_dispatch {
    enabled: true
    arms: ["arm-a", "arm-b"]
    budget: { max_calls_per_request: 2 max_tokens_per_request: 512 reserve_tokens_per_arm: 256 max_concurrency_per_request: 2 max_response_bytes_per_request: 700 }
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	p := &cfg.Decisions[0].Plugins[0]
	if p.Type != "shadow_dispatch" {
		t.Fatalf("expected shadow_dispatch plugin, got %s", p.Type)
	}
	sd, ok := decodePluginConfig[config.ShadowDispatchPluginConfig](p)
	if !ok {
		t.Fatalf("could not decode shadow_dispatch plugin config")
	}
	if len(sd.Arms) != 2 || sd.Arms[0] != "arm-a" || sd.Arms[1] != "arm-b" {
		t.Errorf("arms = %v, want [arm-a arm-b]", sd.Arms)
	}
	if sd.Budget.MaxCallsPerRequest != 2 || sd.Budget.MaxTokensPerRequest != 512 || sd.Budget.ReserveTokensPerArm != 256 || sd.Budget.MaxConcurrencyPerRequest != 2 {
		t.Errorf("budget = %+v, want calls=2 tokens=512 reserve=256 concurrency=2", sd.Budget)
	}

	dsl, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	for _, want := range []string{`"arm-a"`, `"arm-b"`, "max_calls_per_request: 2", "max_tokens_per_request: 512", "reserve_tokens_per_arm: 256", "max_concurrency_per_request: 2", "max_response_bytes_per_request: 700"} {
		if !strings.Contains(dsl, want) {
			t.Errorf("decompiled DSL missing %q in:\n%s", want, dsl)
		}
	}
}

func shadowBudgetDSL(budget string) string {
	return `ROUTE test {
  PRIORITY 1
  WHEN domain("math")
  MODEL "m:1b"
  PLUGIN shadow_dispatch {
    enabled: true
    budget: ` + budget + `
  }
}`
}

func TestCompileShadowDispatchBudgetRejectsMalformed(t *testing.T) {
	cases := []struct {
		name    string
		budget  string
		wantErr string
	}{
		{name: "non-object", budget: `"bad"`, wantErr: "shadow_dispatch.budget must be an object"},
		{name: "string bound", budget: `{ max_calls_per_request: "1" }`, wantErr: "cannot unmarshal string into Go struct field ShadowDispatchBudgetConfig.max_calls_per_request"},
		{name: "bool bound", budget: `{ max_response_bytes_per_request: true }`, wantErr: "cannot unmarshal bool into Go struct field ShadowDispatchBudgetConfig.max_response_bytes_per_request"},
		{name: "string cost", budget: `{ max_cost_per_request: "1" }`, wantErr: "cannot unmarshal string into Go struct field ShadowDispatchBudgetConfig.max_cost_per_request"},
		{name: "unknown field", budget: `{ max_calls: 1 }`, wantErr: `unknown field "max_calls"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, errs := Compile(shadowBudgetDSL(tc.budget))
			if len(errs) == 0 {
				t.Fatalf("budget %s compiled without error, want %q", tc.budget, tc.wantErr)
			}
			if !strings.Contains(errs[0].Error(), tc.wantErr) {
				t.Errorf("errors = %v, want %q", errs, tc.wantErr)
			}
		})
	}
}

func TestCompileShadowDispatchBudgetKeepsExplicitZero(t *testing.T) {
	cfg, errs := Compile(shadowBudgetDSL(`{ max_calls_per_request: 0 max_tokens_per_request: 0 }`))
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	sd, ok := decodePluginConfig[config.ShadowDispatchPluginConfig](&cfg.Decisions[0].Plugins[0])
	if !ok {
		t.Fatal("could not decode shadow_dispatch plugin config")
	}
	if sd.Budget != (config.ShadowDispatchBudgetConfig{}) {
		t.Errorf("budget = %+v, want zero budget", sd.Budget)
	}
}
