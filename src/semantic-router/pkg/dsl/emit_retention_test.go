package dsl

import (
	"encoding/json"
	"strings"
	"testing"
)

const emitRoundtripDSL = `ROUTE clear_context {
  PRIORITY 100
  MODEL "fast-small"
  EMIT retention {
    drop: true
  }
}

ROUTE keep_context {
  PRIORITY 90
  MODEL "quality-large"
  EMIT retention {
    ttl_turns: 3
    keep_current_model: true
    prefer_prefix_retention: true
  }
}
`

func TestCompileEmitRetentionDirectives(t *testing.T) {
	cfg, errs := Compile(emitRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 2 {
		t.Fatalf("expected 2 decisions, got %d", len(cfg.Decisions))
	}

	d0 := cfg.Decisions[0]
	if len(d0.Emits) != 1 || d0.Emits[0].Kind != "retention" {
		t.Fatalf("expected one retention emit on decision[0], got %+v", d0.Emits)
	}
	if d0.Emits[0].Retention == nil || d0.Emits[0].Retention.Drop == nil || !*d0.Emits[0].Retention.Drop {
		t.Fatalf("expected drop=true, got %+v", d0.Emits[0].Retention)
	}

	d1 := cfg.Decisions[1]
	ret := d1.Emits[0].Retention
	if ret == nil {
		t.Fatalf("expected retention directive on decision[1]")
	}
	if ret.TTLTurns == nil || *ret.TTLTurns != 3 {
		t.Fatalf("ttl_turns expected 3, got %v", ret.TTLTurns)
	}
	if ret.KeepCurrentModel == nil || !*ret.KeepCurrentModel {
		t.Fatalf("keep_current_model expected true, got %v", ret.KeepCurrentModel)
	}
	if ret.PreferPrefixRetention == nil || !*ret.PreferPrefixRetention {
		t.Fatalf("prefer_prefix_retention expected true, got %v", ret.PreferPrefixRetention)
	}
}

func TestDecompileEmitRetentionRoundTrip(t *testing.T) {
	cfg, errs := Compile(emitRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	out, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	if !strings.Contains(out, "EMIT retention") || !strings.Contains(out, "drop: true") {
		t.Fatalf("decompiled DSL missing EMIT retention / drop:\n%s", out)
	}
	// Re-compile decompiled output to verify round-trip equivalence.
	cfg2, errs2 := Compile(out)
	if len(errs2) > 0 {
		t.Fatalf("second compile errors: %v\n--- source ---\n%s", errs2, out)
	}
	if len(cfg2.Decisions) != 2 {
		t.Fatalf("round-trip decision count mismatch")
	}
	if cfg2.Decisions[0].Emits[0].Retention.Drop == nil || !*cfg2.Decisions[0].Emits[0].Retention.Drop {
		t.Fatalf("round-trip lost drop=true")
	}
	if cfg2.Decisions[1].Emits[0].Retention.TTLTurns == nil || *cfg2.Decisions[1].Emits[0].Retention.TTLTurns != 3 {
		t.Fatalf("round-trip lost ttl_turns=3")
	}
}

func TestEmitRoutingYAMLIncludesRetention(t *testing.T) {
	yamlBytes, errs := EmitRoutingYAML(emitRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("EmitRoutingYAML errors: %v", errs)
	}
	yamlStr := string(yamlBytes)
	if !strings.Contains(yamlStr, "emits:") {
		t.Fatalf("expected emits: in routing YAML:\n%s", yamlStr)
	}
	if !strings.Contains(yamlStr, "kind: retention") {
		t.Fatalf("expected kind: retention in routing YAML:\n%s", yamlStr)
	}
	if !strings.Contains(yamlStr, "drop: true") {
		t.Fatalf("expected drop: true in routing YAML:\n%s", yamlStr)
	}
	if !strings.Contains(yamlStr, "ttl_turns: 3") {
		t.Fatalf("expected ttl_turns: 3 in routing YAML:\n%s", yamlStr)
	}
}

func TestProgramJSONIncludesEmits(t *testing.T) {
	prog, errs := Parse(emitRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	data, err := MarshalProgramJSON(prog)
	if err != nil {
		t.Fatalf("MarshalProgramJSON: %v", err)
	}
	var out struct {
		Routes []struct {
			Name  string `json:"name"`
			Emits []struct {
				Kind      string `json:"kind"`
				Retention *struct {
					Drop                  *bool `json:"drop"`
					TTLTurns              *int  `json:"ttl_turns"`
					KeepCurrentModel      *bool `json:"keep_current_model"`
					PreferPrefixRetention *bool `json:"prefer_prefix_retention"`
				} `json:"retention"`
			} `json:"emits"`
		} `json:"routes"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal program JSON: %v", err)
	}
	if len(out.Routes) != 2 {
		t.Fatalf("expected 2 routes in JSON, got %d", len(out.Routes))
	}
	if len(out.Routes[0].Emits) != 1 || out.Routes[0].Emits[0].Kind != "retention" {
		t.Fatalf("route[0] missing retention emit: %+v", out.Routes[0].Emits)
	}
	r0 := out.Routes[0].Emits[0].Retention
	if r0 == nil || r0.Drop == nil || !*r0.Drop {
		t.Fatalf("route[0] retention.drop expected true, got %+v", r0)
	}
	r1 := out.Routes[1].Emits[0].Retention
	if r1 == nil || r1.TTLTurns == nil || *r1.TTLTurns != 3 {
		t.Fatalf("route[1] retention.ttl_turns expected 3, got %+v", r1)
	}
}

func TestValidatorEmitRules(t *testing.T) {
	cases := []struct {
		name      string
		input     string
		wantLevel DiagLevel
		wantSub   string
	}{
		{
			name: "unknown_kind",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT bogus {}
}`,
			wantLevel: DiagError,
			wantSub:   "unknown EMIT kind",
		},
		{
			name: "duplicate_kind",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { drop: true }
  EMIT retention { ttl_turns: 2 }
}`,
			wantLevel: DiagError,
			wantSub:   "duplicate EMIT kind",
		},
		{
			name: "drop_ttl_conflict",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { drop: true, ttl_turns: 3 }
}`,
			wantLevel: DiagError,
			wantSub:   "drop=true conflicts with ttl_turns",
		},
		{
			name: "negative_ttl",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { ttl_turns: -1 }
}`,
			wantLevel: DiagError,
			wantSub:   "ttl_turns must be >= 0",
		},
		{
			name: "empty_block_warn",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention {}
}`,
			wantLevel: DiagWarning,
			wantSub:   "empty block",
		},
		{
			name: "zero_ttl_no_drop_warn",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { ttl_turns: 0 }
}`,
			wantLevel: DiagWarning,
			wantSub:   "ttl_turns=0 without drop",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			diags, _ := Validate(tc.input)
			found := false
			for _, d := range diags {
				if d.Level == tc.wantLevel && strings.Contains(d.Message, tc.wantSub) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("expected %s diagnostic containing %q; got:\n%v", tc.wantLevel, tc.wantSub, diags)
			}
		})
	}
}
