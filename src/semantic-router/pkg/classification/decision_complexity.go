package classification

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func prepareDecisionComplexity(models *classifierModelRuntime, rules []config.ComplexityRule) (*ComplexityClassifier, error) {
	if models == nil || models.cfg == nil {
		return nil, nil
	}
	prepared := &ComplexityClassifier{judgments: make(map[string]*decisionJudgment)}
	for _, rule := range rules {
		if models.cfg.ComplexityRuleUsesPrototypes(rule) {
			continue
		}
		if _, err := rule.EffectiveBoundaries(); err != nil {
			return nil, err
		}
		definition, _ := modelservice.BuiltinTask("complexity")
		question := definition.Question
		question.ID = ""
		if description := strings.TrimSpace(rule.Description); description != "" {
			question.Instructions += " " + description
		}
		question.Levels = append([]string(nil), question.Levels...)
		for level, candidates := range map[int][]string{0: rule.Easy.Candidates, 2: rule.Hard.Candidates} {
			var examples []string
			for _, candidate := range candidates {
				if candidate = strings.TrimSpace(candidate); candidate != "" {
					examples = append(examples, candidate)
				}
			}
			if len(examples) > 0 {
				question.Levels[level] += ": " + strings.Join(examples, "; ")
			}
		}
		judgment, err := newDecisionJudgment(models, "complexity", "complexity", &question)
		if err != nil || judgment == nil {
			return nil, err
		}
		question = judgment.plan.Question
		question.ID += ":" + rule.Name
		judgment.plan, err = modelservice.CompileTask(judgment.plan.Definition, question, judgment.card)
		if err != nil {
			return nil, err
		}
		prepared.judgments[rule.Name] = judgment
		prepared.rules = append(prepared.rules, rule)
	}
	if len(prepared.rules) == 0 {
		return nil, nil
	}
	return prepared, nil
}

func (c *ComplexityClassifier) classifyJudgments(ctx context.Context, text string) ([]ComplexityRuleResult, error) {
	var rules []config.ComplexityRule
	for _, rule := range c.rules {
		if c.judgments[rule.Name] != nil {
			rules = append(rules, rule)
		}
	}
	results := make([]ComplexityRuleResult, len(rules))
	errors := make([]error, len(rules))
	modelservice.Fan(ctx, len(rules), func(i int) {
		rule := rules[i]
		answer, err := c.judgments[rule.Name].ask(ctx, modelservice.Request{State: text})
		if err != nil {
			errors[i] = err
			return
		}
		score := answer.Score
		maximum := float64(len(c.judgments[rule.Name].plan.Question.Levels) - 1)
		if maximum <= 0 || math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > maximum {
			errors[i] = fmt.Errorf("complexity returned an invalid score")
			return
		}
		score /= maximum
		// Explicit boundaries read the native [0,1] score. The legacy symmetric
		// threshold is a distance around zero, so map that scale explicitly.
		if rule.HardAbove == nil && rule.EasyBelow == nil && rule.HardBelow == nil && rule.EasyAbove == nil {
			score = 2*score - 1
		}
		bounds, _ := rule.EffectiveBoundaries()
		results[i] = ComplexityRuleResult{RuleName: rule.Name, Difficulty: bounds.Verdict(score), FusedMargin: score, SignalSource: "decision_score"}
	})
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}
