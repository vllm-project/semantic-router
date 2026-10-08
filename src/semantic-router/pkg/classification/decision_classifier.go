package classification

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type decisionLabelClassifier struct {
	judgment    *decisionJudgment
	labels      []string
	independent bool
}

func (j *decisionJudgment) questionDeployment() string        { return j.deployment }
func (c *decisionLabelClassifier) questionDeployment() string { return c.judgment.deployment }
func (c *decisionLabelClassifier) readsWholeText() bool       { return !c.judgment.plan.Question.Truncate }

func prepareDecisionLabelClassifier(models *classifierModelRuntime, rule config.ClassifierSignalRule) (labelClassifier, error) {
	question := modelservice.Question{Type: "choice", Instructions: rule.Instructions, Truncate: true}
	if question.Instructions == "" {
		question.Instructions = "Which category best describes the request?"
	}
	for _, label := range rule.Labels {
		question.Choices = append(question.Choices, modelservice.Choice{Key: label, Description: label})
	}
	judgment, err := newDecisionJudgment(models, "classifier."+rule.Name, "classifier", &question)
	if err != nil || judgment == nil {
		return nil, err
	}
	return &decisionLabelClassifier{judgment: judgment, labels: append([]string(nil), rule.Labels...)}, nil
}

func (c *decisionLabelClassifier) Classify(ctx context.Context, input string) (labelClassification, error) {
	answer, err := c.judgment.ask(ctx, modelservice.Request{State: input})
	if err != nil {
		return labelClassification{}, err
	}
	if c.independent {
		if !classifierScoresFinite(c.labels, answer.Probabilities) {
			return labelClassification{}, fmt.Errorf("incomplete independent judgment scores")
		}
		return labelClassification{Scores: answer.Probabilities}, nil
	}
	scores, err := validateLLMLabelScores(c.labels, answer.Probabilities)
	if err != nil {
		return labelClassification{}, fmt.Errorf("decision classifier distribution: %w", err)
	}
	return labelClassification{Scores: scores}, nil
}

func prepareDecisionSafety(models *classifierModelRuntime, consumer string, labels []string, multiLabel bool) (labelClassifier, error) {
	id, kind, instructions := "safety", "choice", "Is this request harmful?"
	if multiLabel {
		id, kind, instructions = "safety_categories", "set", "Which harmful categories apply to this request?"
	}
	question := modelservice.Question{Type: kind, Instructions: instructions, Truncate: false}
	for _, label := range labels {
		choice := modelservice.Choice{Key: label, Description: label}
		if multiLabel {
			question.Labels = append(question.Labels, choice)
		} else {
			question.Choices = append(question.Choices, choice)
		}
	}
	// Keep the published binary safety question byte-for-byte: its option
	// descriptions are semantic input, not merely display labels.
	if !multiLabel && slices.Equal(labels, []string{"safe", "unsafe"}) {
		definition, _ := modelservice.BuiltinTask("safety")
		question = definition.Question
		question.ID = consumer + ":p_harm"
	}
	judgment, err := newDecisionJudgment(models, consumer, id, &question)
	if err != nil || judgment == nil {
		return nil, err
	}
	return &decisionLabelClassifier{judgment: judgment, labels: labels, independent: multiLabel}, nil
}

func prepareDecisionPreference(models *classifierModelRuntime, rules []config.PreferenceRule) (*decisionJudgment, error) {
	question := modelservice.Question{Type: "choice", Instructions: "Which configured response preference best matches what this user wants?", Truncate: true}
	for _, rule := range rules {
		description := rule.Description
		if len(rule.Examples) > 0 {
			description += " Examples: " + strings.Join(rule.Examples, "; ")
		}
		question.Choices = append(question.Choices, modelservice.Choice{Key: rule.Name, Description: description})
	}
	return newDecisionJudgment(models, "preference", "preference", &question)
}

func (p *PreferenceClassifier) ClassifyContext(ctx context.Context, input string) (*PreferenceResult, error) {
	if p.judgment == nil {
		return p.Classify(input)
	}
	answer, err := p.judgment.ask(ctx, modelservice.Request{State: input})
	if err != nil {
		return nil, err
	}
	labels := make([]string, 0, len(p.preferenceRules))
	for _, rule := range p.preferenceRules {
		labels = append(labels, rule.Name)
	}
	scores, err := validateLLMLabelScores(labels, answer.Probabilities)
	if err != nil {
		return nil, fmt.Errorf("preference distribution: %w", err)
	}
	for _, rule := range p.preferenceRules {
		if answer.Choice == rule.Name {
			confidence := scores[rule.Name]
			if confidence < float64(rule.Threshold) {
				return nil, ErrPreferenceBelowThreshold
			}
			return &PreferenceResult{Preference: rule.Name, Confidence: float32(confidence)}, nil
		}
	}
	return nil, fmt.Errorf("preference answer selected an undeclared rule")
}
