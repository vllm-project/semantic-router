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
    budget: { max_calls_per_request: 2 max_tokens_per_request: 512 reserve_tokens_per_arm: 256 }
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
	if sd.Budget.MaxCallsPerRequest != 2 || sd.Budget.MaxTokensPerRequest != 512 || sd.Budget.ReserveTokensPerArm != 256 {
		t.Errorf("budget = %+v, want calls=2 tokens=512 reserve=256", sd.Budget)
	}

	dsl, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	for _, want := range []string{`"arm-a"`, `"arm-b"`, "max_calls_per_request: 2", "max_tokens_per_request: 512", "reserve_tokens_per_arm: 256"} {
		if !strings.Contains(dsl, want) {
			t.Errorf("decompiled DSL missing %q in:\n%s", want, dsl)
		}
	}
}
