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
	if config.HasImageCandidatesInRules(rules) {
		return nil, nil
	}
	prepared := &ComplexityClassifier{rules: rules, judgments: make(map[string]*decisionJudgment)}
	for _, rule := range rules {
		if _, err := rule.EffectiveBoundaries(); err != nil {
			return nil, err
		}
		question := modelservice.Question{
			Type: "score", Truncate: true,
			Instructions: "How difficult is this request to solve correctly? " + rule.Description,
			Levels:       []string{"easy: " + strings.Join(rule.Easy.Candidates, "; "), "moderate difficulty", "hard: " + strings.Join(rule.Hard.Candidates, "; ")},
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
	}
	return prepared, nil
}

func (c *ComplexityClassifier) classifyJudgments(ctx context.Context, text string) ([]ComplexityRuleResult, error) {
	results := make([]ComplexityRuleResult, len(c.rules))
	errors := make([]error, len(c.rules))
	modelservice.Fan(ctx, len(c.rules), func(i int) {
		rule := c.rules[i]
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
