package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
	return &llmLabelClassifier{
		client:       client,
		model:        external.ModelName,
		labels:       append([]string(nil), rule.Labels...),
		instructions: rule.Instructions,
		timeout:      timeout,
	}, nil
}

func (c *llmLabelClassifier) Classify(
	ctx context.Context,
	input string,
) (labelClassification, error) {
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	systemPrompt := fmt.Sprintf(
		"%s\n\nChoose exactly one label from: %s.\n"+
			`Return only JSON: {"label":"<exact label>","rationale":"<short reason>"}.`,
		strings.TrimSpace(c.instructions),
		strings.Join(c.labels, ", "),
	)
	response, err := c.client.GenerateWithSystemPrompt(
		callCtx,
		c.model,
		systemPrompt,
		input,
		&GenerationOptions{
			MaxTokens:   128,
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
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	if len(raw) != 2 || raw["label"] == nil || raw["rationale"] == nil {
		return labelClassification{}, fmt.Errorf(
			"classifier response must contain exactly label and rationale",
		)
	}
	var choice struct {
		Label     string `json:"label"`
		Rationale string `json:"rationale"`
	}
	if err := json.Unmarshal(
		[]byte(content),
		&choice,
	); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	choice.Label = strings.TrimSpace(choice.Label)
	choice.Rationale = strings.TrimSpace(choice.Rationale)
	if choice.Rationale == "" {
		return labelClassification{}, fmt.Errorf("classifier returned an empty rationale")
	}
	if !containsLabel(c.labels, choice.Label) {
		return labelClassification{}, fmt.Errorf("classifier returned undeclared label %q", choice.Label)
	}
	scores := make(map[string]float64, len(c.labels))
	for _, label := range c.labels {
		if label == choice.Label {
			scores[label] = 1
		} else {
			scores[label] = 0
		}
	}
	return labelClassification{Scores: scores, Rationale: choice.Rationale}, nil
}

func containsLabel(labels []string, target string) bool {
	for _, label := range labels {
		if label == target {
			return true
		}
	}
	return false
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
