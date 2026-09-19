package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

const (
	defaultLLMLabelClassifierMaxTokens = 128
	llmLabelScoreSumTolerance          = 0.02
)

type labelClassification struct {
	Scores    map[string]float64
	Rationale string
	// ScoreWindows is used by explicit native window scans. Each entry is a
	// complete distribution; consumers must not merge class-wise maxima.
	ScoreWindows []map[string]float64
	// Thresholds is non-nil only for a prepared independent-score policy.
	Thresholds  map[string]float64
	PolicyTrace *ClassifierRuleMetrics
}

type labelClassifier interface {
	Classify(context.Context, string) (labelClassification, error)
}

type llmLabelClassifier struct {
	client           *VLLMClient
	model            string
	labels           []string
	instructions     string
	disableRationale bool
	timeout          time.Duration
	maxTokens        int
	reasoning        *reasoningRequestControl
	handle           *binding.Resolved[string, labelClassification]
	recipe           string
}

func newLLMLabelClassifier(
	rule config.ClassifierSignalRule,
	external *config.ExternalModelConfig,
	reasoning *reasoningRequestControl,
	models ...*classifierModelRuntime,
) (labelClassifier, error) {
	if external == nil {
		return nil, fmt.Errorf("external model %q is not configured", rule.Model)
	}
	client := newVLLMClientFromConfig(external)
	if client.initErr != nil {
		return nil, client.initErr
	}
	timeout := time.Duration(external.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	maxTokens := external.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultLLMLabelClassifierMaxTokens
	}
	classifier := &llmLabelClassifier{
		client:           client,
		model:            external.ModelName,
		labels:           append([]string(nil), rule.Labels...),
		instructions:     rule.Instructions,
		disableRationale: rule.DisableRationale,
		timeout:          timeout,
		maxTokens:        maxTokens,
		reasoning:        reasoning,
	}
	runtime := consumerModelRuntime(models)
	backendCfg := &config.RemoteClassifierBackend{Model: external.Name, Protocol: config.RemoteClassifierProtocolHTTPChat, Contract: config.RemoteClassifierContractLabelDistribution}
	spec := runtime.remoteSpec("classifier."+rule.Name, backendCfg)
	handle, err := remoteTaskBinding(context.Background(), runtime, spec, external, client, classifier.classify, func(_ string, out labelClassification) error {
		_, err := validateLLMLabelScores(classifier.labels, out.Scores)
		return err
	})
	if err != nil {
		return nil, err
	}
	classifier.handle, classifier.recipe = handle, string(spec.Recipe)
	return classifier, nil
}

func (c *llmLabelClassifier) Classify(ctx context.Context, input string) (labelClassification, error) {
	return c.handle.Call(ctx, c.recipe, input)
}
func (c *llmLabelClassifier) Close() error { return c.handle.Close() }

func (c *llmLabelClassifier) classify(
	ctx context.Context,
	input string,
) (labelClassification, error) {
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	responseFields := `"scores" and "rationale"`
	rationaleInstruction := ` "rationale" must be a short reason.`
	if c.disableRationale {
		responseFields = `"scores"`
		rationaleInstruction = ""
	}
	systemPrompt := fmt.Sprintf(
		"%s\n\nScore every label from: %s.\n"+
			`Return only JSON with exactly %s. `+
			`"scores" must map every exact label to a number between 0 and 1, `+
			`and all scores must sum to 1.%s`,
		strings.TrimSpace(c.instructions),
		strings.Join(c.labels, ", "),
		responseFields,
		rationaleInstruction,
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
			reasoning:   c.reasoning,
		},
	)
	if err != nil {
		return labelClassification{}, err
	}
	if len(response.Choices) == 0 {
		return labelClassification{}, fmt.Errorf("classifier returned no choices")
	}
	content := strings.TrimSpace(response.Choices[0].Message.Content)
	return parseLLMLabelClassification(content, c.labels, c.disableRationale)
}

func parseLLMLabelClassification(
	content string,
	labels []string,
	disableRationale bool,
) (labelClassification, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	if disableRationale {
		if raw["scores"] == nil || len(raw) > 2 || (len(raw) == 2 && raw["rationale"] == nil) {
			return labelClassification{}, fmt.Errorf(
				"classifier response must contain exactly scores and optional rationale",
			)
		}
	} else if len(raw) != 2 || raw["scores"] == nil || raw["rationale"] == nil {
		return labelClassification{}, fmt.Errorf(
			"classifier response must contain exactly scores and rationale",
		)
	}
	var result struct {
		Scores    map[string]*float64 `json:"scores"`
		Rationale *string             `json:"rationale"`
	}
	if err := json.Unmarshal(
		[]byte(content),
		&result,
	); err != nil {
		return labelClassification{}, fmt.Errorf("classifier returned invalid JSON: %w", err)
	}
	rationale := ""
	if result.Rationale != nil {
		rationale = strings.TrimSpace(*result.Rationale)
	} else if disableRationale && raw["rationale"] != nil {
		return labelClassification{}, fmt.Errorf("classifier rationale must be a string")
	}
	if !disableRationale && rationale == "" {
		return labelClassification{}, fmt.Errorf("classifier returned an empty rationale")
	}
	reported := make(map[string]float64, len(result.Scores))
	for label, score := range result.Scores {
		if score == nil {
			return labelClassification{}, fmt.Errorf("classifier score for label %q must be a number", label)
		}
		reported[label] = *score
	}
	scores, err := validateLLMLabelScores(labels, reported)
	if err != nil {
		return labelClassification{}, err
	}
	return labelClassification{Scores: scores, Rationale: rationale}, nil
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
			classifier, err = newLocalLabelClassifier(rule, b.models)
		case config.ClassifierSignalTypeLLM:
			external := b.cfg.FindExternalModelByName(rule.Model)
			reasoning, reasoningErr := resolveExternalModelReasoningControl(
				b.cfg.ReasoningFamilies,
				external,
			)
			if reasoningErr != nil {
				err = fmt.Errorf("external model %q: %w", rule.Model, reasoningErr)
				break
			}
			classifier, err = newLLMLabelClassifier(
				rule,
				external,
				reasoning,
				b.models,
			)
		case config.ClassifierSignalTypeSequenceClassifier:
			classifier, err = newSequenceLabelClassifier(
				rule,
				b.cfg.FindExternalModelByName(rule.Model),
				b.models,
			)
		default:
			// Config validation rejects unknown types, so reaching here means a
			// rule bypassed it. Fail instead of storing a nil classifier that
			// silently drops the signal at request time.
			err = fmt.Errorf("unsupported type %q", rule.Type)
		}
		if err != nil {
			closeLabelClassifiers(classifiers)
			return nil, fmt.Errorf("build classifier signal %q: %w", rule.Name, err)
		}
		classifiers[rule.Name] = classifier
	}
	return withGenericClassifiers(classifiers), nil
}

func closeLabelClassifiers(classifiers map[string]labelClassifier) {
	for _, classifier := range classifiers {
		if closer, ok := classifier.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}
}
