package systemone

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// acceptNative applies only operator-authored, uncalibrated predicates. A raw
// model statistic is not a probability of correctness. Rules are conjunctive,
// and a gate that judges the whole bundle (the quality policy's, or a stage's
// without one) must cover every answer without dropping failures.
func acceptNative(observations []Observation, accept *config.EscalateAccept, whole bool) bool {
	if accept == nil || len(accept.Native) == 0 || len(observations) == 0 {
		return false
	}
	covered := make([]bool, len(observations))
	for _, rule := range accept.Native {
		matched := false
		for i, observation := range observations {
			if !ruleMatches(rule, observation.Question) {
				continue
			}
			matched, covered[i] = true, true
			value, known := observationStatistic(observation, rule.Field)
			if !known || !predicateMatches(value, rule.NumericPredicate) {
				return false
			}
		}
		if !matched && (rule.Question != "" || rule.State != nil) {
			return false
		}
	}
	for i, observation := range observations {
		if !observation.Valid || (whole && !covered[i]) {
			return false
		}
	}
	return true
}

func ruleMatches(rule config.NativeAcceptanceRule, question Question) bool {
	return (rule.State == nil || *rule.State == question.State) &&
		(rule.Question == "" || rule.Question == question.Name) &&
		(rule.QuestionType == "" || rule.QuestionType == question.Type)
}

func observationStatistic(o Observation, field string) (float64, bool) {
	if !o.Valid {
		return 0, false
	}
	switch field {
	case "top_probability":
		if len(o.Distribution) > 0 {
			return o.Distribution[0], true
		}
	case "confidence":
		if o.Confidence != nil {
			return *o.Confidence, true
		}
	case "probability_margin":
		if o.Type == "noul" && o.Probability != nil {
			return 2 * math.Abs(*o.Probability-0.5), true
		}
	}
	return 0, false
}

func predicateMatches(value float64, p config.NumericPredicate) bool {
	return finite(value) && (p.GT == nil || value > *p.GT) &&
		(p.GTE == nil || value >= *p.GTE) &&
		(p.LT == nil || value < *p.LT) && (p.LTE == nil || value <= *p.LTE)
}
