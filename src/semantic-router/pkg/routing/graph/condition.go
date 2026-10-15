package graph

import (
	"context"
	"encoding/json"
	"regexp"
	"slices"
)

// Condition is what a branch case or a loop's Until tests.
type Condition interface {
	Holds(ctx context.Context, x *Exec, st *State) (bool, error)
}

// ConditionFunc adapts a function to Condition.
type ConditionFunc func(ctx context.Context, x *Exec, st *State) (bool, error)

// Holds calls f.
func (f ConditionFunc) Holds(ctx context.Context, x *Exec, st *State) (bool, error) {
	return f(ctx, x, st)
}

// Succeeded holds when the latest step left results and all of them
// succeeded.
func Succeeded() Condition {
	return ConditionFunc(func(_ context.Context, _ *Exec, st *State) (bool, error) {
		if len(st.Results) == 0 {
			return false, nil
		}
		for _, result := range st.Results {
			if !result.OK() {
				return false, nil
			}
		}
		return true, nil
	})
}

// ContentMatches holds when the first result's text matches pattern.
func ContentMatches(pattern *regexp.Regexp) Condition {
	return ConditionFunc(func(_ context.Context, _ *Exec, st *State) (bool, error) {
		return len(st.Results) > 0 && pattern.MatchString(st.Results[0].Content()), nil
	})
}

// SignalMatched holds when the request matched the named signal of
// signalType, as the decision engine's signal conditions read it.
func SignalMatched(signalType, name string) Condition {
	return ConditionFunc(func(_ context.Context, x *Exec, _ *State) (bool, error) {
		return slices.Contains(x.Signals(signalType), name), nil
	})
}

// ScoreAtLeast holds when the number under key is at least threshold, such as a
// score an aggregate step recorded.
func ScoreAtLeast(key Key[float64], threshold float64) Condition {
	return ConditionFunc(func(_ context.Context, _ *Exec, st *State) (bool, error) {
		value, ok := key.Get(st)
		return ok && value >= threshold, nil
	})
}

// LogprobAtLeast holds when the first result's average token log
// probability is at least threshold: the model's own confidence in its
// answer. A result without log probabilities never holds.
func LogprobAtLeast(threshold float64) Condition {
	return ConditionFunc(func(_ context.Context, _ *Exec, st *State) (bool, error) {
		if len(st.Results) == 0 {
			return false, nil
		}
		average, ok := averageLogprob(st.Results[0].Body)
		return ok && average >= threshold, nil
	})
}

// All holds when every condition holds.
func All(conditions ...Condition) Condition {
	return ConditionFunc(func(ctx context.Context, x *Exec, st *State) (bool, error) {
		for _, c := range conditions {
			if holds, err := c.Holds(ctx, x, st); err != nil || !holds {
				return false, err
			}
		}
		return true, nil
	})
}

// Any holds when some condition holds.
func Any(conditions ...Condition) Condition {
	return ConditionFunc(func(ctx context.Context, x *Exec, st *State) (bool, error) {
		for _, c := range conditions {
			if holds, err := c.Holds(ctx, x, st); err != nil || holds {
				return holds, err
			}
		}
		return false, nil
	})
}

// Not holds when c does not.
func Not(c Condition) Condition {
	return ConditionFunc(func(ctx context.Context, x *Exec, st *State) (bool, error) {
		holds, err := c.Holds(ctx, x, st)
		return !holds && err == nil, err
	})
}

func averageLogprob(body []byte) (float64, bool) {
	var completion struct {
		Choices []struct {
			Logprobs struct {
				Content []struct {
					Logprob float64 `json:"logprob"`
				} `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &completion) != nil || len(completion.Choices) == 0 {
		return 0, false
	}
	tokens := completion.Choices[0].Logprobs.Content
	if len(tokens) == 0 {
		return 0, false
	}
	var sum float64
	for _, token := range tokens {
		sum += token.Logprob
	}
	return sum / float64(len(tokens)), true
}
