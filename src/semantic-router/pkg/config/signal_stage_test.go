package config

import "testing"

func TestSignalStageOf(t *testing.T) {
	if got := SignalStageOf(SignalTypeJailbreak); got != SignalStageRequest {
		t.Errorf("jailbreak stage = %q, want %q", got, SignalStageRequest)
	}
	if got := SignalStageOf(SignalTypeResponseJailbreak); got != SignalStageResponse {
		t.Errorf("response_jailbreak stage = %q, want %q", got, SignalStageResponse)
	}
	// An unknown type is request-stage rather than an error, so adding a signal
	// type does not silently make its decisions unroutable.
	if got := SignalStageOf("something_new"); got != SignalStageRequest {
		t.Errorf("unknown type stage = %q, want %q", got, SignalStageRequest)
	}
}

func TestDecisionStageFindsAResponseSignalAtAnyDepth(t *testing.T) {
	requestOnly := RuleNode{
		Operator: "AND",
		Conditions: []RuleNode{
			{Type: SignalTypeDomain, Name: "business"},
			{Type: SignalTypeJailbreak, Name: "prompt_injection"},
		},
	}
	if got := DecisionStage(&requestOnly); got != SignalStageRequest {
		t.Errorf("request-only rules = %q, want %q", got, SignalStageRequest)
	}

	nested := RuleNode{
		Operator: "AND",
		Conditions: []RuleNode{
			{Type: SignalTypeDomain, Name: "business"},
			{
				Operator: "OR",
				Conditions: []RuleNode{
					{Type: SignalTypeKeyword, Name: "code_keywords"},
					{Type: SignalTypeResponseJailbreak, Name: "unsafe_completion"},
				},
			},
		},
	}
	if got := DecisionStage(&nested); got != SignalStageResponse {
		t.Errorf("nested response signal = %q, want %q", got, SignalStageResponse)
	}

	if got := DecisionStage(nil); got != SignalStageRequest {
		t.Errorf("nil rules = %q, want %q", got, SignalStageRequest)
	}
}

// A configuration whose decisions all wait on the model's output has nothing
// left to pick a model with, so it must not load.
func TestValidateDecisionStagesRequiresARequestStageDecision(t *testing.T) {
	responseOnly := &RouterConfig{}
	responseOnly.Decisions = []Decision{
		{
			Name:  "block_unsafe_output",
			Rules: RuleNode{Type: SignalTypeResponseJailbreak, Name: "unsafe_completion"},
		},
	}
	if err := validateDecisionStages(responseOnly); err == nil {
		t.Fatal("a configuration with only response-stage decisions must not validate")
	}

	mixed := &RouterConfig{}
	mixed.Decisions = []Decision{
		{
			Name:  "block_unsafe_output",
			Rules: RuleNode{Type: SignalTypeResponseJailbreak, Name: "unsafe_completion"},
		},
		{
			Name:  "business",
			Rules: RuleNode{Type: SignalTypeDomain, Name: "business"},
		},
	}
	if err := validateDecisionStages(mixed); err != nil {
		t.Fatalf("a configuration with one request-stage decision must validate: %v", err)
	}

	empty := &RouterConfig{}
	if err := validateDecisionStages(empty); err != nil {
		t.Fatalf("no decisions is another validator's problem: %v", err)
	}
}

// #3205 was about jailbreak rules that shipped unvalidated. A response-stage
// rule that can never match, or that matches everything, has the same shape of
// failure: the decision reading it silently does the wrong thing.
func TestValidateResponseJailbreakRules(t *testing.T) {
	withRules := func(rules ...ResponseJailbreakRule) *RouterConfig {
		cfg := &RouterConfig{}
		cfg.Signals.ResponseJailbreakRules = rules
		return cfg
	}

	if err := validateResponseJailbreakRules(withRules(
		ResponseJailbreakRule{Name: "unsafe_completion", Threshold: 0.7},
	)); err != nil {
		t.Fatalf("a well formed rule must validate: %v", err)
	}

	cases := map[string]ResponseJailbreakRule{
		"missing name":        {Threshold: 0.7},
		"omitted threshold":   {Name: "unsafe_completion"},
		"zero threshold":      {Name: "unsafe_completion", Threshold: 0},
		"threshold above one": {Name: "unsafe_completion", Threshold: 1.5},
	}
	for name, rule := range cases {
		if err := validateResponseJailbreakRules(withRules(rule)); err == nil {
			t.Errorf("%s must be rejected", name)
		}
	}

	dup := withRules(
		ResponseJailbreakRule{Name: "unsafe_completion", Threshold: 0.7},
		ResponseJailbreakRule{Name: "unsafe_completion", Threshold: 0.8},
	)
	if err := validateResponseJailbreakRules(dup); err == nil {
		t.Error("duplicate rule names must be rejected")
	}
}
