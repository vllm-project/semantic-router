package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// A generic rule owns a prepared sequence binding, independent of every other
// rule and generation. Only provider-compatible physical weights are shared.
type localLabelClassifier struct {
	labels  []string
	backend *ownedSequenceBackend
}

func newLocalLabelClassifier(rule config.ClassifierSignalRule, models ...*classifierModelRuntime) (labelClassifier, error) {
	return newLocalLabelClassifierForBinding("classifier."+rule.Name, rule, models...)
}

func newLocalLabelClassifierForBinding(consumerName string, rule config.ClassifierSignalRule, models ...*classifierModelRuntime) (labelClassifier, error) {
	runtime := consumerModelRuntime(models)
	spec := runtime.localSpec(consumerName, rule.ModelPath, "auto", config.RemoteClassifierContractLabelDistribution, rule.UseCPU)
	if spec.Binding.Contract == config.RemoteClassifierContractLabelScores {
		handle, err := runtime.runtime.OperatingPoint(context.Background(), spec, rule.Labels)
		if err != nil {
			return nil, fmt.Errorf("initialize classifier %q: %w", rule.Name, err)
		}
		return &localOperatingPointClassifier{handle: handle, recipe: string(spec.Recipe), labels: append([]string(nil), rule.Labels...)}, nil
	}
	backend := &ownedSequenceBackend{runtime: runtime.runtime, spec: spec, labels: append([]string(nil), rule.Labels...)}
	if err := backend.Init(rule.ModelPath, rule.UseCPU, len(rule.Labels)); err != nil {
		return nil, fmt.Errorf("initialize classifier %q: %w", rule.Name, err)
	}
	return &localLabelClassifier{labels: append([]string(nil), rule.Labels...), backend: backend}, nil
}

func (c *localLabelClassifier) Classify(ctx context.Context, input string) (labelClassification, error) {
	result, err := c.backend.Classify(ctx, input)
	if err != nil {
		return labelClassification{}, err
	}
	if len(result.Probabilities) != len(c.labels) {
		return labelClassification{}, fmt.Errorf("model returned %d probabilities for %d labels", len(result.Probabilities), len(c.labels))
	}
	scores := make(map[string]float64, len(c.labels))
	for index, label := range c.labels {
		scores[label] = float64(result.Probabilities[index])
	}
	return labelClassification{Scores: scores}, nil
}
func (c *localLabelClassifier) Close() error { return c.backend.Close() }

type localOperatingPointClassifier struct {
	handle *native.OperatingPointScorer
	recipe string
	labels []string
}

func (c *localOperatingPointClassifier) Classify(ctx context.Context, input string) (labelClassification, error) {
	output, err := c.handle.Score(ctx, c.recipe, input)
	if err != nil {
		return labelClassification{}, err
	}
	scores, err := namedLabelScores(c.labels, output.Scores)
	if err != nil {
		return labelClassification{}, err
	}
	thresholds, err := namedLabelScores(c.labels, c.handle.Thresholds())
	if err != nil {
		return labelClassification{}, err
	}
	capability := c.handle.Capability()
	trace := &ClassifierRuleMetrics{PolicySHA256: c.handle.PolicySHA256(), Provider: capability.Provider, Device: capability.Device, Precision: capability.Precision, Windows: output.Windows, InputTokens: output.Input.OriginalTokens, ProcessedTokens: output.Input.ProcessedTokens, Truncated: output.Input.Truncated, Thresholds: thresholds, WindowBatchSize: 1}
	return labelClassification{Scores: scores, Thresholds: thresholds, PolicyTrace: trace}, nil
}
func (c *localOperatingPointClassifier) Close() error      { return c.handle.Close() }
func (*localOperatingPointClassifier) ownsAdmission() bool { return true }
