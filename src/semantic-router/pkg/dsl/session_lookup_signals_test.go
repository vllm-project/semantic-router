package dsl

import "testing"

func TestParseCompileSessionMetricRules(t *testing.T) {
	src := `SESSION_STATE session_routing {
  turn_number: int
  current_model: string
  candidate_model: string
  cumulative_cost_usd: float
}

SIGNAL session_metric session_cost_pressure {
  kind: "state"
  state: "session_routing.cumulative_cost_usd"
  normalize: "minmax"
  min: 0
  max: 10
}

SIGNAL session_metric handoff_penalty {
  kind: "lookup"
  table: "handoff_penalties"
  key: ["session_routing.current_model", "session_routing.candidate_model"]
}
`
	cfg, errs := Compile(src)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.SessionMetricRules) != 2 {
		t.Fatalf("session_metrics: %+v", cfg.SessionMetricRules)
	}
	if cfg.SessionMetricRules[0].Name != "session_cost_pressure" || cfg.SessionMetricRules[0].Kind != "state" {
		t.Fatalf("first rule: %+v", cfg.SessionMetricRules[0])
	}
	if cfg.SessionMetricRules[0].State != "session_routing.cumulative_cost_usd" {
		t.Fatalf("state: %q", cfg.SessionMetricRules[0].State)
	}
	if cfg.SessionMetricRules[1].Name != "handoff_penalty" || cfg.SessionMetricRules[1].Kind != "lookup" {
		t.Fatalf("second rule: %+v", cfg.SessionMetricRules[1])
	}
	if len(cfg.SessionMetricRules[1].Key) != 2 {
		t.Fatalf("lookup key: %+v", cfg.SessionMetricRules[1].Key)
	}
}

func TestValidateSessionRuleRequiresNumericState(t *testing.T) {
	src := `SESSION_STATE session_routing {
  current_model: string
}

SIGNAL session_metric bad {
  kind: "state"
  state: "session_routing.current_model"
}
`
	diags, _ := Validate(src)
	if len(diags) == 0 {
		t.Fatalf("expected diagnostics for non-numeric state")
	}
}

func TestValidateLookupKeyMustBeDeclaredSessionState(t *testing.T) {
	src := `SESSION_STATE session_routing {
  current_model: string
}

SIGNAL session_metric bad {
  kind: "lookup"
  table: "handoff_penalties"
  key: ["session_routing.not_a_field"]
}
`
	diags, _ := Validate(src)
	if len(diags) == 0 {
		t.Fatalf("expected lookup key diagnostic, got %#v", diags)
	}
}

func TestProjectionScoreMayReferenceSessionMetricInputs(t *testing.T) {
	src := `SESSION_STATE session_routing {
  turn_number: int
  current_model: string
  candidate_model: string
  cumulative_cost_usd: float
}

SIGNAL session_metric session_cost_pressure {
  kind: "state"
  state: "session_routing.cumulative_cost_usd"
}

SIGNAL session_metric handoff_penalty {
  kind: "lookup"
  table: "handoff_penalties"
  key: ["session_routing.current_model", "session_routing.candidate_model"]
}

PROJECTION score blended {
  method: "weighted_sum"
  inputs: [
    { type: "session_metric", name: "session_cost_pressure", weight: 1.0, value_source: "raw" },
    { type: "session_metric", name: "handoff_penalty", weight: 0.5, value_source: "raw" }
  ]
}
`
	diags, _ := Validate(src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics: %#v", diags)
	}
}
