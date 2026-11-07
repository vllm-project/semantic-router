package classification

import "testing"

// questionAnswers matches the named decision-model questions and conversation
// facts it lists; each name in unknown is a question whose answer failed. They
// are controlled inputs to the shipped rules, not model output.
func questionAnswers(questions, facts, unknown []string) *SignalResults {
	in := &SignalResults{MatchedDecisionRules: questions, MatchedConversationRules: facts, SignalErrors: map[string]string{}}
	for _, name := range unknown {
		in.SignalErrors["decision:"+name] = "unavailable"
	}
	return in
}

func TestBuiltinSpeedAndCostAskForToolUseAndDepth(t *testing.T) {
	for _, profile := range []struct{ name, fallback string }{{"speed", "fast"}, {"cost", "economy"}} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			for _, tt := range []struct {
				name string
				in   *SignalResults
				want string
			}{
				{"plain request", questionAnswers(nil, nil, nil), profile.fallback},
				{"tools merely available", questionAnswers(nil, []string{"has_tools"}, nil), profile.fallback},
				{"asked to call a declared tool", questionAnswers([]string{"tool_use"}, []string{"has_tools"}, nil), "tools"},
				{"tool request without definitions", questionAnswers([]string{"tool_use"}, nil, nil), profile.fallback},
				{"required tool", questionAnswers(nil, []string{"has_tools", "tool_required"}, nil), "tools"},
				{"tools disabled by the client", questionAnswers([]string{"tool_use"}, []string{"has_tools", "tool_disabled"}, nil), profile.fallback},
				{"active tool loop", questionAnswers(nil, []string{"tool_loop"}, nil), "tools"},
				{"depth outranks tools", questionAnswers([]string{"depth", "tool_use"}, []string{"has_tools"}, nil), "reasoning"},
				{"unknown depth keeps quality", questionAnswers(nil, nil, []string{"depth"}), "reasoning"},
				{"unknown tool question falls back", questionAnswers(nil, []string{"has_tools"}, []string{"tool_use"}), profile.fallback},
			} {
				t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, tt.in, tt.want) })
			}
		})
	}
	c := builtinPolicyClassifier(t, "cost")
	for _, tt := range []struct {
		name string
		in   *SignalResults
		want string
	}{
		{"reported wrong answer", questionAnswers([]string{"correction"}, nil, nil), "reasoning"},
		{"correction given as text to translate", questionAnswers([]string{"correction", "quoted"}, nil, nil), "economy"},
	} {
		t.Run("cost/"+tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, tt.in, tt.want) })
	}
}

func TestBuiltinAccuracyAsksWhoDoesTheWork(t *testing.T) {
	c := builtinPolicyClassifier(t, "accuracy")
	for _, tt := range []struct {
		name string
		in   *SignalResults
		want string
	}{
		{"one direct answer", questionAnswers([]string{"mode:one"}, nil, nil), "simple"},
		{"several workers", questionAnswers([]string{"mode:workers"}, nil, nil), "agent"},
		{"several workers as text to translate", questionAnswers([]string{"mode:workers", "quoted"}, nil, nil), "simple"},
		{"several workers during a tool loop", questionAnswers([]string{"mode:workers"}, []string{"tool_loop"}, nil), "reasoning"},
		{"resumed workflow", questionAnswers(nil, []string{"flow"}, nil), "agent"},
		{"independent experts", questionAnswers([]string{"mode:experts"}, nil, nil), "review"},
		{"independent experts with a required tool", questionAnswers([]string{"mode:experts"}, []string{"tool_required"}, nil), "reasoning"},
		{"hard problem", questionAnswers([]string{"mode:one", "depth"}, nil, nil), "reasoning"},
		{"fact check", questionAnswers([]string{"verification"}, nil, nil), "reasoning"},
		{"consequential question", questionAnswers([]string{"stakes"}, nil, nil), "reasoning"},
		{"reported wrong answer", questionAnswers([]string{"correction"}, nil, nil), "reasoning"},
		{"hard phrase to define", questionAnswers([]string{"depth", "verification", "quoted"}, nil, nil), "simple"},
		{"unknown depth keeps quality", questionAnswers(nil, nil, []string{"depth"}), "reasoning"},
		{"unknown mode never fans out", questionAnswers(nil, nil, []string{"mode"}), "simple"},
	} {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, tt.in, tt.want) })
	}
}

func TestBuiltinVaultQuestionsFailClosed(t *testing.T) {
	c := builtinPolicyClassifier(t, "vault")
	for _, tt := range []struct {
		name string
		in   *SignalResults
		want string
	}{
		{"ordinary private request", questionAnswers(nil, nil, nil), "private"},
		{"personal data in the conversation", &SignalResults{MatchedPIIRules: []string{"personal_data"}}, "sensitive"},
		{"confidential material", questionAnswers([]string{"confidential"}, nil, nil), "sensitive"},
		{"personal data about a person", questionAnswers([]string{"personal_data"}, nil, nil), "sensitive"},
		{"risk of harm", questionAnswers([]string{"risk"}, nil, nil), "sensitive"},
		{"personal advice", questionAnswers([]string{"personal_advice"}, nil, nil), "sensitive"},
		{"prompt attack outranks sensitivity", &SignalResults{MatchedPIIRules: []string{"personal_data"}, MatchedJailbreakRules: []string{"prompt_attack"}}, "guard"},
		{"unavailable prompt guard", &SignalResults{SignalErrors: map[string]string{"jailbreak:prompt_attack": "unavailable"}}, ""},
		{"unavailable PII judgment", &SignalResults{SignalErrors: map[string]string{"pii:personal_data": "unavailable"}}, ""},
		{"unanswered sensitivity question", questionAnswers(nil, nil, []string{"risk"}), ""},
	} {
		t.Run(tt.name, func(t *testing.T) { assertBuiltinPolicy(t, c, tt.in, tt.want) })
	}
}
