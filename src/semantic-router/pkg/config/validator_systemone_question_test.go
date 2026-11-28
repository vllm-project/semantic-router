package config

import (
	"strings"
	"testing"
)

// A blank choice description and repeated score levels load as written, and
// the runtime refuses the question at request time, so validation does.
func TestSystemOneQuestionRefusesWhatTheRuntimeRefuses(t *testing.T) {
	for name, test := range map[string]struct {
		question SystemOneQuestion
		want     string
	}{
		"blank choice description": {
			question: SystemOneQuestion{Type: SystemOneQuestionChoice, Instructions: "Which?", Choices: []SystemOneOption{{Key: "a", Description: "  "}, {Key: "b"}}},
			want:     "blank description",
		},
		"repeated score level": {
			question: SystemOneQuestion{Type: SystemOneQuestionScore, Instructions: "How hard?", Levels: []string{"Easy", " Easy", "Hard"}},
			want:     "must differ",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := validateSystemOneQuestion(test.question); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validateSystemOneQuestion = %v, want an error naming %q", err, test.want)
			}
		})
	}
	valid := SystemOneQuestion{Type: SystemOneQuestionChoice, Instructions: "Which?", Choices: []SystemOneOption{{Key: "a"}, {Key: "b", Description: "the second"}}}
	if err := validateSystemOneQuestion(valid); err != nil {
		t.Fatalf("a choice with and without descriptions was refused: %v", err)
	}
}
