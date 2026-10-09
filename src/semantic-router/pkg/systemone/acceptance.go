package systemone

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// acceptNative applies only operator-authored, uncalibrated predicates. A raw
// model statistic is not a probability of correctness. Rules are conjunctive,
// and a whole-bundle gate must cover every answer without dropping failures.
func acceptNative(observations []Observation, acceptance *config.NativeAcceptance, whole bool) bool {
	if acceptance == nil || len(acceptance.Rules) == 0 || len(observations) == 0 {
		return false
	}
	covered := make([]bool, len(observations))
	for _, rule := range acceptance.Rules {
		matched := false
		for i, observation := range observations {
			if !ruleMatches(rule, observation.Question) {
				continue
			}
			matched, covered[i] = true, true
			value, known := observationStatistic(observation, rule.Field)
			if !known || !predicateMatches(value, rule.Predicate) {
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

// FeatureNames is a versioned, order-sensitive contract shared with the
// research collector. Changing a feature requires a new artifact version.
var FeatureNames = []string{
	"bias", "min_top_probability", "mean_top_probability", "max_entropy",
	"min_margin", "log_state_bytes", "log_questions", "fraction_choice",
	"fraction_noul", "fraction_score", "invalid_response",
}

// Features describes the complete bundle. Unknown distributions set the
// invalid-response feature and contribute conservative missing evidence.
func (r *NativeRequest) Features(observations []Observation) []float64 {
	f := []float64{1, 1, 0, 0, 1, math.Log1p(float64(r.StateBytes)), math.Log1p(float64(len(r.Questions))), 0, 0, 0, 0}
	if len(observations) == 0 {
		f[1], f[3], f[4], f[10] = 0, 1, 0, 1
		return f
	}
	for _, o := range observations {
		switch o.Type {
		case "choice":
			f[7]++
		case "noul":
			f[8]++
		case "score":
			f[9]++
		}
		if !o.Valid || len(o.Distribution) < 2 {
			f[1], f[3], f[4], f[10] = 0, 1, 0, 1
			continue
		}
		top := o.Distribution[0]
		f[1], f[2] = math.Min(f[1], top), f[2]+top
		f[4] = math.Min(f[4], top-o.Distribution[1])
		entropy := 0.0
		for _, p := range o.Distribution {
			if p > 0 {
				entropy -= p * math.Log(p)
			}
		}
		f[3] = math.Max(f[3], entropy/math.Log(float64(len(o.Distribution))))
	}
	f[2] /= float64(len(observations))
	for i := 7; i <= 9; i++ {
		f[i] /= float64(len(observations))
	}
	return f
}
