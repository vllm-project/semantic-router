package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const defaultLLMLabelClassifierMaxTokens = 128
const llmLabelScoreSumTolerance = 0.02

type labelClassification struct {
	Scores    map[string]float64
	Rationale string
}

type labelClassifier interface {
	Classify(context.Context, string) (labelClassification, error)
}

type llmLabelClassifier struct {
	client       *VLLMClient
	model        string
	labels       []string
	instructions string
	timeout      time.Duration
	maxTokens    int
}

func newLLMLabelClassifier(
	rule config.ClassifierSignalRule,
	external *config.ExternalModelConfig,
) (labelClassifier, error) {
	if external == nil {
		return nil, fmt.Errorf("external model %q is not configured", rule.Model)
	}
	client := newVLLMClientFromConfig(external)
	timeout := time.Duration(external.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	maxTokens := external.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultLLMLabelClassifierMaxTokens
	}
	return &llmLabelClassifier{
		client:       client,
		model:        external.ModelName,
		labels:       append([]string(nil), rule.Labels...),
		instructions: rule.Instructions,
		timeout:      timeout,
		maxTokens:    maxTokens,
	}, nil
}

func (c *llmLabelClassifier) Classify(
	ctx context.Context,
	input string,
) (labelClassification, error) {
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	systemPrompt := fmt.Sprintf(
		"%s\n\nScore every label from: %s.\n"+
			`Return only JSON with exactly "scores" and "rationale". `+
			`"scores" must map every exact label to a number between 0 and 1, `+
			`and all scores must sum to 1. "rationale" must be a short reason.`,
		strings.TrimSpace(c.instructions),
		strings.Join(c.labels, ", "),
	)
	response, err := c.client.GenerateWithSystemPrompt(
		callCtx,
		c.model,
		systemPrompt,
		input,
		&GenerationOptions{
			MaxTokens:   c.maxTokens,
			Temperature: 0,
			JSONMode:    true,
		},
	)
	if err != nil {
		return labelClassification{}, err
	}
	if len(response.Choices) == 0 {
		return labelClassification{}, fmt.Errorf("classifier returned no choices")
	}
	content := strings.TrimSpace(response.Choices[0].Message.Content)
	return parseLLMLabelClassification(content, c.labels)
}

func parseLLMLabelClassification(
	content string,
	labels []string,
) (labelClassification, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	if len(raw) != 2 || raw["scores"] == nil || raw["rationale"] == nil {
		return labelClassification{}, fmt.Errorf(
			"classifier response must contain exactly scores and rationale",
		)
	}
	var result struct {
		Scores    map[string]float64 `json:"scores"`
		Rationale string             `json:"rationale"`
	}
	if err := json.Unmarshal(
		[]byte(content),
		&result,
	); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	result.Rationale = strings.TrimSpace(result.Rationale)
	if result.Rationale == "" {
		return labelClassification{}, fmt.Errorf("classifier returned an empty rationale")
	}
	scores, err := validateLLMLabelScores(labels, result.Scores)
	if err != nil {
		return labelClassification{}, err
	}
	return labelClassification{Scores: scores, Rationale: result.Rationale}, nil
}

func validateLLMLabelScores(
	labels []string,
	reported map[string]float64,
) (map[string]float64, error) {
	if len(reported) != len(labels) {
		return nil, fmt.Errorf("classifier scores must contain exactly the declared labels")
	}
	scores := make(map[string]float64, len(labels))
	var sum float64
	for _, label := range labels {
		score, ok := reported[label]
		if !ok {
			return nil, fmt.Errorf("classifier scores are missing label %q", label)
		}
		if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
			return nil, fmt.Errorf(
				"classifier score for label %q must be finite and within [0, 1]",
				label,
			)
		}
		scores[label] = score
		sum += score
	}
	if math.Abs(sum-1) > llmLabelScoreSumTolerance {
		return nil, fmt.Errorf("classifier scores sum to %v, want approximately 1", sum)
	}
	return scores, nil
}

func withGenericClassifiers(classifiers map[string]labelClassifier) option {
	return func(c *Classifier) {
		c.genericClassifiers = classifiers
	}
}

func (b *classifierOptionBuilder) buildGenericClassifiersOption() (option, error) {
	if len(b.cfg.ClassifierRules) == 0 {
		return nil, nil
	}
	classifiers := make(map[string]labelClassifier, len(b.cfg.ClassifierRules))
	for _, rule := range b.cfg.ClassifierRules {
		var (
			classifier labelClassifier
			err        error
		)
		switch rule.Type {
		case config.ClassifierSignalTypeLocal:
			classifier, err = newLocalLabelClassifier(rule)
		case config.ClassifierSignalTypeLLM:
			classifier, err = newLLMLabelClassifier(
				rule,
				b.cfg.FindExternalModelByName(rule.Model),
			)
		case config.ClassifierSignalTypeSequenceClassifier:
			classifier, err = newSequenceLabelClassifier(
				rule,
				b.cfg.FindExternalModelByName(rule.Model),
			)
		default:
			// Config validation rejects unknown types, so reaching here means a
			// rule bypassed it. Fail instead of storing a nil classifier that
			// silently drops the signal at request time.
			err = fmt.Errorf("unsupported type %q", rule.Type)
		}
		if err != nil {
			return nil, fmt.Errorf("build classifier signal %q: %w", rule.Name, err)
		}
		classifiers[rule.Name] = classifier
	}
	return withGenericClassifiers(classifiers), nil
}
