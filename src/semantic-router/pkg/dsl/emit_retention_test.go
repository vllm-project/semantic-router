package dsl

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	assertDecisionCount(t, cfg.Decisions, 2)

	assertConfigRetentionDrop(t, cfg.Decisions[0].Emits[0].Retention, true, "decision[0]")
	assertConfigRetentionTTLTurns(t, cfg.Decisions[1].Emits[0].Retention, 3, "decision[1]")
	assertConfigRetentionKeepCurrentModel(t, cfg.Decisions[1].Emits[0].Retention, true, "decision[1]")
	assertConfigRetentionPreferPrefix(t, cfg.Decisions[1].Emits[0].Retention, true, "decision[1]")
}

func assertDecisionCount(t *testing.T, decisions []config.Decision, want int) {
	t.Helper()
	if len(decisions) != want {
		t.Fatalf("expected %d decisions, got %d", want, len(decisions))
	}
	for i, decision := range decisions {
		if len(decision.Emits) != 1 || decision.Emits[0].Kind != "retention" {
			t.Fatalf("decision[%d] expected one retention emit, got %+v", i, decision.Emits)
		}
	}
}

func assertConfigRetentionDrop(t *testing.T, r *config.RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.Drop == nil || *r.Drop != want {
		t.Fatalf("%s retention.drop = %v, want %v", context, retentionBoolValue(r, "drop"), want)
	}
}

func assertConfigRetentionTTLTurns(t *testing.T, r *config.RetentionDirective, want int, context string) {
	t.Helper()
	if r == nil || r.TTLTurns == nil || *r.TTLTurns != want {
		t.Fatalf("%s retention.ttl_turns = %v, want %d", context, retentionIntValue(r), want)
	}
}

func assertConfigRetentionKeepCurrentModel(t *testing.T, r *config.RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.KeepCurrentModel == nil || *r.KeepCurrentModel != want {
		t.Fatalf("%s retention.keep_current_model = %v, want %v", context, retentionBoolValue(r, "keep_current_model"), want)
	}
}

func assertConfigRetentionPreferPrefix(t *testing.T, r *config.RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.PreferPrefixRetention == nil || *r.PreferPrefixRetention != want {
		t.Fatalf("%s retention.prefer_prefix_retention = %v, want %v", context, retentionBoolValue(r, "prefer_prefix_retention"), want)
	}
}

func retentionBoolValue(r *config.RetentionDirective, field string) interface{} {
	if r == nil {
		return nil
	}
	switch field {
	case "drop":
		return r.Drop
	case "keep_current_model":
		return r.KeepCurrentModel
	case "prefer_prefix_retention":
		return r.PreferPrefixRetention
	default:
		return nil
	}
}

func retentionIntValue(r *config.RetentionDirective) interface{} {
	if r == nil {
		return nil
	}
	return r.TTLTurns
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

func TestDecisionTreeEmitRetentionLoweringAndRoundTrip(t *testing.T) {
	prog, cfg := compileDecisionTreeEmitRetention(t)
	assertDecisionTreeASTRetention(t, prog)
	assertDecisionTreeConfigRetention(t, cfg)
	assertDecisionTreeRetentionRoundTrip(t, cfg)
}

func compileDecisionTreeEmitRetention(t *testing.T) (*Program, *config.RouterConfig) {
	t.Helper()
	input := `SIGNAL keyword fast { keywords: ["fast"] }
SIGNAL keyword hard { keywords: ["hard"] }

DECISION_TREE retention_policy {
  IF keyword("fast") {
    NAME "fast_route"
    MODEL "fast-small"
    EMIT retention { drop: true }
  }
  ELSE IF keyword("hard") {
    NAME "hard_route"
    MODEL "quality-large"
    EMIT retention {
      ttl_turns: 2
      keep_current_model: true
    }
  }
  ELSE {
    NAME "fallback_route"
    MODEL "balanced"
    EMIT retention { prefer_prefix_retention: true }
  }
}`

	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	cfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		t.Fatalf("compile errors: %v", compileErrs)
	}
	return prog, cfg
}

func assertDecisionTreeASTRetention(t *testing.T, prog *Program) {
	t.Helper()
	if len(prog.Routes) != 3 {
		t.Fatalf("expected 3 lowered routes, got %d", len(prog.Routes))
	}
	assertRouteEmit(t, prog.Routes[0], "IF branch")
	assertRouteEmit(t, prog.Routes[1], "ELSE IF branch")
	assertRouteEmit(t, prog.Routes[2], "ELSE branch")
	assertDSLRetentionDrop(t, prog.Routes[0].Emits[0].Retention, true, "IF branch")
	assertDSLRetentionTTLTurns(t, prog.Routes[1].Emits[0].Retention, 2, "ELSE IF branch")
	assertDSLRetentionKeepCurrentModel(t, prog.Routes[1].Emits[0].Retention, true, "ELSE IF branch")
	assertDSLRetentionPreferPrefix(t, prog.Routes[2].Emits[0].Retention, true, "ELSE branch")
}

func assertRouteEmit(t *testing.T, route *RouteDecl, context string) {
	t.Helper()
	if len(route.Emits) != 1 || route.Emits[0].Kind != "retention" {
		t.Fatalf("%s route %q emits = %+v, want one retention emit", context, route.Name, route.Emits)
	}
}

func assertDSLRetentionDrop(t *testing.T, r *RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.Drop == nil || *r.Drop != want {
		t.Fatalf("%s lost drop=%v", context, want)
	}
}

func assertDSLRetentionTTLTurns(t *testing.T, r *RetentionDirective, want int, context string) {
	t.Helper()
	if r == nil || r.TTLTurns == nil || *r.TTLTurns != want {
		t.Fatalf("%s lost ttl_turns=%d", context, want)
	}
}

func assertDSLRetentionKeepCurrentModel(t *testing.T, r *RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.KeepCurrentModel == nil || *r.KeepCurrentModel != want {
		t.Fatalf("%s lost keep_current_model=%v", context, want)
	}
}

func assertDSLRetentionPreferPrefix(t *testing.T, r *RetentionDirective, want bool, context string) {
	t.Helper()
	if r == nil || r.PreferPrefixRetention == nil || *r.PreferPrefixRetention != want {
		t.Fatalf("%s lost prefer_prefix_retention=%v", context, want)
	}
}

func assertDecisionTreeConfigRetention(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	assertDecisionCount(t, cfg.Decisions, 3)
	assertConfigRetentionDrop(t, cfg.Decisions[0].Emits[0].Retention, true, "compiled IF branch")
	assertConfigRetentionTTLTurns(t, cfg.Decisions[1].Emits[0].Retention, 2, "compiled ELSE IF branch")
	assertConfigRetentionPreferPrefix(t, cfg.Decisions[2].Emits[0].Retention, true, "compiled ELSE branch")
}

func assertDecisionTreeRetentionRoundTrip(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	out, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	cfg2, errs := Compile(out)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v\n--- source ---\n%s", errs, out)
	}
	assertDecisionCount(t, cfg2.Decisions, 3)
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
	out := marshalEmitProgramJSON(t)
	if len(out.Routes) != 2 {
		t.Fatalf("expected 2 routes in JSON, got %d", len(out.Routes))
	}
	assertJSONRouteRetentionDrop(t, out.Routes[0].Emits, true, "route[0]")
	assertJSONRouteRetentionTTLTurns(t, out.Routes[1].Emits, 3, "route[1]")
}

type emitProgramJSON struct {
	Routes []emitRouteJSON `json:"routes"`
}

type emitRouteJSON struct {
	Name  string         `json:"name"`
	Emits []emitEmitJSON `json:"emits"`
}

type emitEmitJSON struct {
	Kind      string             `json:"kind"`
	Retention *emitRetentionJSON `json:"retention"`
}

type emitRetentionJSON struct {
	Drop                  *bool `json:"drop"`
	TTLTurns              *int  `json:"ttl_turns"`
	KeepCurrentModel      *bool `json:"keep_current_model"`
	PreferPrefixRetention *bool `json:"prefer_prefix_retention"`
}

func marshalEmitProgramJSON(t *testing.T) emitProgramJSON {
	t.Helper()
	prog, errs := Parse(emitRoundtripDSL)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	data, err := MarshalProgramJSON(prog)
	if err != nil {
		t.Fatalf("MarshalProgramJSON: %v", err)
	}
	var out emitProgramJSON
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal program JSON: %v", err)
	}
	return out
}

func assertJSONRouteRetentionDrop(t *testing.T, emits []emitEmitJSON, want bool, context string) {
	t.Helper()
	r := requireJSONRetention(t, emits, context)
	if r.Drop == nil || *r.Drop != want {
		t.Fatalf("%s retention.drop expected %v, got %+v", context, want, r)
	}
}

func assertJSONRouteRetentionTTLTurns(t *testing.T, emits []emitEmitJSON, want int, context string) {
	t.Helper()
	r := requireJSONRetention(t, emits, context)
	if r.TTLTurns == nil || *r.TTLTurns != want {
		t.Fatalf("%s retention.ttl_turns expected %d, got %+v", context, want, r)
	}
}

func requireJSONRetention(t *testing.T, emits []emitEmitJSON, context string) *emitRetentionJSON {
	t.Helper()
	if len(emits) != 1 || emits[0].Kind != "retention" || emits[0].Retention == nil {
		t.Fatalf("%s missing retention emit: %+v", context, emits)
	}
	return emits[0].Retention
}

var validatorEmitRuleCases = []struct {
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
	{
		name: "unknown_retention_field",
		input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { mystery: true }
}`,
		wantLevel: DiagError,
		wantSub:   "unknown field",
	},
	{
		name: "wrong_drop_type",
		input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { drop: "yes" }
}`,
		wantLevel: DiagError,
		wantSub:   `field "drop" must be bool`,
	},
	{
		name: "wrong_ttl_type",
		input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { ttl_turns: "2" }
}`,
		wantLevel: DiagError,
		wantSub:   `field "ttl_turns" must be int`,
	},
}

func TestValidatorEmitRules(t *testing.T) {
	for _, tc := range validatorEmitRuleCases {
		t.Run(tc.name, func(t *testing.T) {
			assertEmitDiagnostic(t, tc.input, tc.wantLevel, tc.wantSub)
		})
	}
}

func assertEmitDiagnostic(t *testing.T, input string, wantLevel DiagLevel, wantSub string) {
	t.Helper()
	diags, _ := Validate(input)
	for _, d := range diags {
		if d.Level == wantLevel && strings.Contains(d.Message, wantSub) {
			return
		}
	}
	t.Fatalf("expected %s diagnostic containing %q; got:\n%v", wantLevel, wantSub, diags)
}

func TestCompileRejectsInvalidEmitRetentionDirectives(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		wantSub string
	}{
		{
			name: "unknown_kind",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT bogus {}
}`,
			wantSub: "unknown EMIT kind",
		},
		{
			name: "unknown_retention_field",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { mystery: true }
}`,
			wantSub: "unknown field",
		},
		{
			name: "wrong_retention_type",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { drop: "yes" }
}`,
			wantSub: `field "drop" must be bool`,
		},
		{
			name: "drop_ttl_conflict",
			input: `ROUTE r {
  PRIORITY 1
  MODEL "m"
  EMIT retention { drop: true, ttl_turns: 3 }
}`,
			wantSub: "drop=true conflicts with ttl_turns",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, errs := Compile(tc.input)
			if len(errs) == 0 {
				t.Fatalf("expected compile error containing %q, got cfg=%+v", tc.wantSub, cfg)
			}
			for _, err := range errs {
				if strings.Contains(err.Error(), tc.wantSub) {
					return
				}
			}
			t.Fatalf("expected compile error containing %q, got %v", tc.wantSub, errs)
		})
	}
}
