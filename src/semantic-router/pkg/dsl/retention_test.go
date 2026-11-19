package dsl

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

const retentionRoundtripDSL = `ROUTE clear_context {
  PRIORITY 100
  MODEL "fast-small"
  RETENTION {
    drop: true
  }
}

ROUTE keep_context {
  PRIORITY 90
  MODEL "quality-large"
  RETENTION {
    ttl_turns: 3
    keep_current_model: true
    prefer_prefix_retention: true
  }
}
`

// A RETENTION block compiles to the decision's retention attribute, field by
// field, and decompiles back to the same block.
func TestCompileRetentionRoundTrip(t *testing.T) {
	cfg, errs := Compile(retentionRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Decisions) != 2 {
		t.Fatalf("decisions %d", len(cfg.Decisions))
	}
	clear, keep := cfg.Decisions[0].Retention, cfg.Decisions[1].Retention
	if clear == nil || clear.Drop == nil || !*clear.Drop || clear.TTLTurns != nil {
		t.Fatalf("clear_context retention %+v", clear)
	}
	if keep == nil || keep.TTLTurns == nil || *keep.TTLTurns != 3 || !*keep.KeepCurrentModel || !*keep.PreferPrefixRetention || keep.Drop != nil {
		t.Fatalf("keep_context retention %+v", keep)
	}
	out, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	if !strings.Contains(out, "RETENTION {\n    drop: true\n  }") || strings.Contains(out, "EMIT") {
		t.Fatalf("decompiled DSL:\n%s", out)
	}
	again, errs := Compile(out)
	if len(errs) > 0 || *again.Decisions[1].Retention.TTLTurns != 3 {
		t.Fatalf("round trip: %v %+v", errs, again.Decisions[1].Retention)
	}
	yamlBytes, errs := EmitRoutingYAML(retentionRoundtripDSL)
	if len(errs) > 0 || !strings.Contains(string(yamlBytes), "retention:") || strings.Contains(string(yamlBytes), "emits") {
		t.Fatalf("routing YAML %v:\n%s", errs, yamlBytes)
	}
}

func TestDecisionTreeRetention(t *testing.T) {
	input := `SIGNAL keyword fast { keywords: ["fast"] }

DECISION_TREE retention_policy {
  IF keyword("fast") {
    NAME "fast_route"
    MODEL "fast-small"
    RETENTION { drop: true }
  }
  ELSE {
    NAME "default_route"
    MODEL "quality-large"
    RETENTION { prefer_prefix_retention: true }
  }
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if retention := cfg.Decisions[0].Retention; retention == nil || !*retention.Drop {
		t.Fatalf("fast_route retention %+v", retention)
	}
	if retention := cfg.Decisions[1].Retention; retention == nil || !*retention.PreferPrefixRetention {
		t.Fatalf("default_route retention %+v", retention)
	}
}

func TestProgramJSONIncludesRetention(t *testing.T) {
	prog, errs := Parse(retentionRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	data, err := json.Marshal(ProgramToJSON(prog))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"retention":{"drop":true`) || strings.Contains(string(data), "emits") {
		t.Fatalf("program JSON: %s", data)
	}
}

// The DSL refuses the cache aliases and shadow_dispatch, naming what replaced
// them.
func TestCompileRefusesRemovedCacheAndShadowForms(t *testing.T) {
	for input, want := range map[string]string{
		`ROUTE r { PRIORITY 1 PLUGIN semantic_cache { enabled: true } MODEL "m" }`:                            `plugin "semantic_cache" was removed; write response_cache`,
		`ROUTE r { PRIORITY 1 PLUGIN semantic-cache { enabled: true } MODEL "m" }`:                            `plugin "semantic-cache" was removed; write response_cache`,
		`ROUTE r { PRIORITY 1 PLUGIN shadow_dispatch { enabled: true, model: "c" } MODEL "m" }`:               `write a shadow block in the decision's router_replay plugin`,
		`ROUTE r { PRIORITY 1 PLUGIN response_cache { enabled: true, similarity_threshold: 0.9 } MODEL "m" }`: `unknown field "similarity_threshold"`,
	} {
		_, errs := Compile(input)
		if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), want) {
			t.Errorf("%s: errors %v, want %q", input, errs, want)
		}
	}
}

// The validator and the compiler hold a RETENTION block to the same rules.
func TestRetentionRules(t *testing.T) {
	route := func(block string) string {
		return "ROUTE r {\n  PRIORITY 1\n  MODEL \"m\"\n  " + block + "\n}\n"
	}
	cases := []struct {
		name      string
		block     string
		wantLevel DiagLevel
		wantSub   string
		compiles  bool
	}{
		{name: "drop with ttl", block: "RETENTION { drop: true, ttl_turns: 3 }", wantLevel: DiagError, wantSub: "ROUTE r RETENTION: drop=true conflicts with ttl_turns=3"},
		{name: "negative ttl", block: "RETENTION { ttl_turns: -1 }", wantLevel: DiagError, wantSub: "ttl_turns must be >= 0"},
		{name: "unknown field", block: "RETENTION { mystery: true }", wantLevel: DiagError, wantSub: `unknown field "mystery"`},
		{name: "drop as a string", block: `RETENTION { drop: "yes" }`, wantLevel: DiagError, wantSub: `field "drop" must be bool, got string`},
		{name: "ttl as a string", block: `RETENTION { ttl_turns: "2" }`, wantLevel: DiagError, wantSub: `field "ttl_turns" must be int, got string`},
		{name: "empty", block: "RETENTION {}", wantLevel: DiagWarning, wantSub: "empty block has no effect", compiles: true},
		{name: "zero ttl without drop", block: "RETENTION { ttl_turns: 0 }", wantLevel: DiagWarning, wantSub: "likely a no-op", compiles: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			diags, _ := Validate(route(tc.block))
			found := false
			for _, d := range diags {
				found = found || (d.Level == tc.wantLevel && strings.Contains(d.Message, tc.wantSub))
			}
			if !found {
				t.Fatalf("expected %s diagnostic containing %q; got:\n%v", tc.wantLevel, tc.wantSub, diags)
			}
			_, errs := Compile(route(tc.block))
			if tc.compiles != (len(errs) == 0) {
				t.Fatalf("compile errors %v, want compiles=%v", errs, tc.compiles)
			}
		})
	}
}
