package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type ownedModalityClassifier struct {
	handle *binding.Resolved[string, tasks.LabelDistribution]
	recipe string
	labels []string
}

func (m *ownedModalityClassifier) Close() error {
	if m == nil || m.handle == nil {
		return nil
	}
	return m.handle.Close()
}

func (m *ownedModalityClassifier) Classify(ctx context.Context, text string) (ModalityClassificationResult, error) {
	if m == nil || m.handle == nil {
		return ModalityClassificationResult{}, fmt.Errorf("modality classifier was not prepared")
	}
	result, err := m.handle.Call(ctx, m.recipe, text)
	if err != nil {
		return ModalityClassificationResult{}, err
	}
	index, confidence := deriveArgmax(result.Probabilities)
	if index < 0 || index >= len(m.labels) {
		return ModalityClassificationResult{}, fmt.Errorf("modality output does not match prepared labels")
	}
	return ModalityClassificationResult{Modality: m.labels[index], Confidence: confidence, ConfidenceAvailable: true, Method: "classifier"}, nil
}

func (b *classifierOptionBuilder) buildModalityClassifierOption() (option, error) {
	md := b.cfg.ModalityDetector
	if !md.Enabled || md.GetMethod() == config.ModalityDetectionKeyword {
		return nil, nil
	}
	models := consumerModelRuntime([]*classifierModelRuntime{b.models})
	_, explicit := models.plan.Lookup(models.recipe, "modality_detector")
	if !explicit && (md.Classifier == nil || md.Classifier.ModelPath == "") {
		return nil, nil
	}
	path, useCPU := "", true
	if md.Classifier != nil {
		path, useCPU = md.Classifier.ModelPath, md.Classifier.UseCPU
	}
	spec := models.localSpec("modality_detector", path, "mmbert32k", config.RemoteClassifierContractLabelDistribution, useCPU)
	handle, err := models.runtime.Sequence(context.Background(), spec)
	if err != nil {
		if md.GetMethod() == config.ModalityDetectionHybrid && !explicit {
			logging.Warnf("Modality classifier preparation failed; using configured keyword fallback: %v", err)
			return nil, nil
		}
		return nil, fmt.Errorf("prepare modality classifier: %w", err)
	}
	labels, err := modalityLabels(handle.Capability().Labels)
	if err != nil {
		_ = handle.Close()
		return nil, err
	}
	classifier := &ownedModalityClassifier{handle: handle, recipe: string(spec.Recipe), labels: labels}
	return func(c *Classifier) { c.modalityInference = classifier }, nil
}

func modalityLabels(labels []string) ([]string, error) {
	if len(labels) != 3 {
		return nil, fmt.Errorf("modality classifier requires three labels")
	}
	result := make([]string, len(labels))
	seen := make(map[string]bool, len(labels))
	for i, label := range labels {
		label = strings.ToUpper(strings.TrimSpace(label))
		if (label != "AR" && label != "DIFFUSION" && label != "BOTH") || seen[label] {
			return nil, fmt.Errorf("unsupported modality classifier labels %v", labels)
		}
		result[i], seen[label] = label, true
	}
	return result, nil
}
