package classification

import (
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func nativeClassResult(result candle_binding.ClassResult, err error) (tasks.ClassResult, error) {
	return tasks.ClassResult{
		Class: result.Class, Confidence: result.Confidence,
		Categories: append([]string(nil), result.Categories...),
	}, err
}

func nativeClassResultWithProbs(result candle_binding.ClassResultWithProbs, err error) (tasks.ClassResultWithProbs, error) {
	return tasks.ClassResultWithProbs{
		Class: result.Class, Confidence: result.Confidence, NumClasses: result.NumClasses,
		Probabilities: append([]float32(nil), result.Probabilities...),
	}, err
}

func nativeTokenEntities(entities []candle_binding.TokenEntity) []tasks.TokenEntity {
	if entities == nil {
		return nil
	}
	result := make([]tasks.TokenEntity, len(entities))
	for i, entity := range entities {
		result[i] = tasks.TokenEntity{
			EntityType: entity.EntityType, Start: entity.Start, End: entity.End,
			Text: entity.Text, Confidence: entity.Confidence,
		}
	}
	return result
}

func nativeTokenResult(result candle_binding.TokenClassificationResult, err error) (tasks.TokenClassificationResult, error) {
	scoresAvailable := true
	return tasks.TokenClassificationResult{Entities: nativeTokenEntities(result.Entities), ScoresAvailable: &scoresAvailable}, err
}

func nativeEmbeddingResult(result *candle_binding.EmbeddingOutput, err error) (*tasks.EmbeddingResult, error) {
	if result == nil {
		return nil, err
	}
	return &tasks.EmbeddingResult{
		Embedding: append([]float32(nil), result.Embedding...), ModelType: result.ModelType,
		SequenceLength: result.SequenceLength, ProcessingTimeMs: result.ProcessingTimeMs,
	}, err
}
