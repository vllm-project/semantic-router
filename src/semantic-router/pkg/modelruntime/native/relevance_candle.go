package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func (r *Runtime) candleRelevance(ctx context.Context, spec config.ResolvedModelBinding, selection config.PairScorerSelection) (*preparedRelevance, error) {
	prepared := &preparedRelevance{}
	options := candleOptions(spec)
	options.ModelType = "" // The dedicated loader validates actual ModernBERT config.
	if spec.Binding.Head != "" {
		return nil, fmt.Errorf("%w: Candle pair heads must reside in the model artifact", binding.ErrCapability)
	}
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	execution, err := json.Marshal(struct {
		Options   candle.InstanceOptions
		Selection config.PairScorerSelection
	}{options, selection})
	if err != nil {
		return nil, err
	}
	id := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: revision, Provider: "candle", Device: options.Device, Precision: options.Precision, Execution: "relevance:" + string(execution)}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, id, budget, gate, func(context.Context) (io.Closer, error) {
		model, loadErr := candle.LoadPairScorer(options, candle.PairScorerSelection{Layer: selection.Layer, Dimension: selection.Dimension})
		if loadErr != nil {
			return nil, nativeError(loadErr)
		}
		return model, nil
	})
	if err != nil {
		return nil, err
	}
	err = resource.Use(ctx, func(value io.Closer) error {
		info, infoErr := value.(*candle.PairScorer).Info()
		if infoErr != nil {
			return nativeError(infoErr)
		}
		if info.PairScorer == nil {
			return fmt.Errorf("%w: missing actual pair scorer selection", binding.ErrCapability)
		}
		prepared.selection = config.PairScorerSelection{Layer: info.PairScorer.Layer, Dimension: info.PairScorer.Dimension}
		prepared.capability = candleCapability(spec, info)
		prepared.execution = []relevanceExecution{{Provider: "candle", Device: info.Device, Precision: info.Precision, MathContract: "candle.modernbert.pair_scores.v1"}}
		return nil
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	prepared.infer = func(_ context.Context, value io.Closer, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
		input := make([]candle.TextPair, len(pairs))
		for i, pair := range pairs {
			input[i] = candle.TextPair{Query: pair.Query, Document: pair.Document}
		}
		output, inferErr := value.(*candle.PairScorer).ScorePairs(input)
		result := tasks.RelevanceScores{Scores: output.Scores, Inputs: make([]tasks.InputUsage, len(output.Inputs))}
		for i, usage := range output.Inputs {
			result.Inputs[i] = *candleInputUsage(usage)
		}
		return result, nativeError(inferErr)
	}
	prepared.resource = resource
	return prepared, nil
}
