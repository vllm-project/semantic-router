package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// effortProjectionClassifier scores effort from a score question's expected
// level and one label of a set question, and routes only on the mapping.
func effortProjectionClassifier() *Classifier {
	return &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{SystemOneRules: []config.SystemOneRule{
				{Name: "difficulty", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionScore, Instructions: "How hard?", Levels: []string{"easy", "hard"}}},
				{Name: "needs", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionSet, Instructions: "What does it need?", Labels: []config.SystemOneOption{{Key: "deliberation"}, {Key: "tools"}}}},
			}},
			Projections: config.Projections{
				Scores: []config.ProjectionScore{{
					Name:   "effort",
					Method: "weighted_sum",
					Inputs: []config.ProjectionScoreInput{
						{Type: config.SignalTypeSystemOne, Name: "difficulty", Weight: 0.5, ValueSource: "raw"},
						{Type: config.SignalTypeSystemOne, Name: "needs:deliberation", Weight: 0.4, ValueSource: "raw"},
					},
				}},
				Mappings: []config.ProjectionMapping{{
					Name: "effort_band", Source: "effort", Method: "threshold_bands",
					Outputs: []config.ProjectionMappingOutput{{Name: "effort_high", GTE: float64Ptr(0.5)}},
				}},
			},
			Decisions: []config.Decision{{
				Name:  "hard",
				Rules: config.RuleCombination{Type: config.SignalTypeProjection, Name: "effort_high"},
			}},
		},
	}}
}

func TestALabelledDecisionProjectionInputAsksItsQuestion(t *testing.T) {
	used := effortProjectionClassifier().getUsedSignals()
	for _, key := range []string{"systemone:difficulty", "systemone:needs"} {
		if !signalRuleUsed(used, config.SignalTypeSystemOne, key[len("systemone:"):]) {
			t.Fatalf("%s feeds the effort projection, so it must be asked; used = %v", key, used)
		}
	}
}

func TestALabelledDecisionProjectionInputReadsThatLabelsProbability(t *testing.T) {
	results := &SignalResults{SignalValues: map[string]float64{
		"systemone:difficulty":         0.8,
		"systemone:needs":              0.9,
		"systemone:needs:deliberation": 0.6,
		"systemone:needs:tools":        0.9,
	}}
	got := effortProjectionClassifier().applyProjections(results)
	if score := got.ProjectionScores["effort"]; math.Abs(score-0.64) > 1e-9 {
		t.Fatalf("effort = %v, want 0.5*0.8 + 0.4*0.6 from the deliberation label, not the set's highest label", score)
	}
	if len(got.MatchedProjectionRules) != 1 || got.MatchedProjectionRules[0] != "effort_high" {
		t.Fatalf("matched projections = %v", got.MatchedProjectionRules)
	}
}
