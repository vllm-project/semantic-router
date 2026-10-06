package classification

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// The independent-score adapter shares the HTTP operation and label alignment,
// while its typed binding cannot masquerade as a categorical distribution.
type remoteSafetyScores struct {
	handle *binding.Resolved[string, tasks.LabelScores]
	recipe string
	labels []string
}

func (c *remoteSafetyScores) Classify(ctx context.Context, text string) (labelClassification, error) {
	out, err := c.handle.Call(ctx, c.recipe, text)
	if err != nil {
		return labelClassification{}, err
	}
	scores, err := namedLabelScores(c.labels, out.Scores)
	return labelClassification{Scores: scores}, err
}
func (c *remoteSafetyScores) Close() error { return c.handle.Close() }

func prepareRemoteSafetyClassifier(models *classifierModelRuntime, spec config.ResolvedModelBinding, external *config.ExternalModelConfig, labels []string, multiLabel bool) (labelClassifier, error) {
	transport, transportErr := NewHTTPClassifierInference(external, newDeclaredLabelMapping(labels))
	if transportErr != nil {
		return nil, transportErr
	}
	if !multiLabel {
		owned, err := prepareRemoteSequence(models, spec, external, transport)
		if err != nil {
			return nil, err
		}
		return &sequenceLabelClassifier{backend: owned, labels: append([]string(nil), labels...)}, nil
	}
	handle, err := remoteTaskBinding(context.Background(), models, spec, external, transport,
		func(ctx context.Context, text string) (tasks.LabelScores, error) {
			wire, err := transport.fetchLabelScores(ctx, text)
			if err != nil {
				return tasks.LabelScores{}, err
			}
			vector, err := alignIndependentLabelScores(transport.mapping, wire)
			return tasks.LabelScores{Scores: vector}, err
		}, func(_ string, out tasks.LabelScores) error { return tasks.ValidateLabelScores(out.Scores) })
	if err != nil {
		return nil, err
	}
	return &remoteSafetyScores{handle: handle, recipe: string(spec.Recipe), labels: append([]string(nil), labels...)}, nil
}
