package classification

import (
	"context"
	"fmt"
	"strings"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var localClassifierLifecycle struct {
	sync.Mutex
	signature string
}

type localLabelClassifier struct {
	labels []string
}

func newLocalLabelClassifier(rule config.ClassifierSignalRule) (labelClassifier, error) {
	modelPath := config.ResolveModelPath(rule.ModelPath)
	signature := fmt.Sprintf("%s|%t|%s", modelPath, rule.UseCPU, strings.Join(rule.Labels, "\x00"))

	localClassifierLifecycle.Lock()
	defer localClassifierLifecycle.Unlock()
	switch {
	case localClassifierLifecycle.signature == "":
		if err := candle_binding.InitGenericClassifier(
			modelPath,
			len(rule.Labels),
			rule.UseCPU,
		); err != nil {
			return nil, fmt.Errorf("initialize classifier %q: %w", rule.Name, err)
		}
		localClassifierLifecycle.signature = signature
	case localClassifierLifecycle.signature != signature:
		return nil, fmt.Errorf(
			"local classifier model or labels changed; restart the router to apply the new classifier",
		)
	}
	return &localLabelClassifier{labels: append([]string(nil), rule.Labels...)}, nil
}

func (c *localLabelClassifier) Classify(
	_ context.Context,
	input string,
) (labelClassification, error) {
	result, err := candle_binding.ClassifyTextWithProbabilities(input)
	if err != nil {
		return labelClassification{}, err
	}
	if result.Class < 0 || result.Class >= len(c.labels) {
		return labelClassification{}, fmt.Errorf(
			"class index %d is outside %d labels",
			result.Class,
			len(c.labels),
		)
	}
	scores := make(map[string]float64, len(c.labels))
	for index, label := range c.labels {
		switch {
		case index < len(result.Probabilities):
			scores[label] = float64(result.Probabilities[index])
		case index == result.Class:
			scores[label] = float64(result.Confidence)
		default:
			scores[label] = 0
		}
	}
	return labelClassification{Scores: scores}, nil
}
