package classification

import (
	"errors"
	"reflect"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestNativeTaskResultsPreserveDistributionAndOwnership(t *testing.T) {
	native := candle_binding.ClassResultWithProbs{
		Class: 2, Confidence: 0.7, Probabilities: []float32{0.1, 0.2, 0.7}, NumClasses: 3,
	}
	result, err := nativeClassResultWithProbs(native, nil)
	if err != nil || result.Class != 2 || result.Confidence != 0.7 || result.NumClasses != 3 {
		t.Fatalf("lost native result fields: %+v, %v", result, err)
	}
	native.Probabilities[0] = 0.9
	if !reflect.DeepEqual(result.Probabilities, []float32{0.1, 0.2, 0.7}) {
		t.Fatalf("task result aliases native result memory: %v", result.Probabilities)
	}
	partial, err := nativeClassResultWithProbs(candle_binding.ClassResultWithProbs{Class: 1, Confidence: 0.6}, nil)
	if err != nil || partial.Probabilities != nil || partial.NumClasses != 0 {
		t.Fatalf("top-1 result acquired a fabricated distribution: %+v, %v", partial, err)
	}
}

func TestNativeTaskResultsKeepByteOffsetsAndPartialError(t *testing.T) {
	text := "前文🙂 unsupported"
	start := len("前文🙂 ")
	native := candle_binding.TokenClassificationResult{Entities: []candle_binding.TokenEntity{{
		EntityType: "unsupported", Start: start, End: len(text), Text: "unsupported", Confidence: 0.75,
	}}}
	wantErr := errors.New("partial native result")
	result, err := nativeTokenResult(native, wantErr)
	if !errors.Is(err, wantErr) || len(result.Entities) != 1 || !result.HasScores() {
		t.Fatalf("partial result or error lost: %+v, %v", result, err)
	}
	span := result.Entities[0]
	if span.Start != start || span.End != len(text) || text[span.Start:span.End] != span.Text || span.Confidence != 0.75 {
		t.Fatalf("span semantics changed: %+v", span)
	}
	native.Entities[0].EntityType = "changed"
	if result.Entities[0].EntityType != "unsupported" {
		t.Fatal("task span aliases native result memory")
	}
	if result.TruncatedAt != nil {
		t.Fatal("adapter invented a native truncation offset")
	}
}

func TestNativeTaskResultsEmbeddingMetadataAndOwnership(t *testing.T) {
	native := &candle_binding.EmbeddingOutput{
		Embedding: []float32{3, 4}, ModelType: "mmbert", SequenceLength: 5, ProcessingTimeMs: 1.25,
	}
	result, err := nativeEmbeddingResult(native, nil)
	if err != nil || result.ModelType != "mmbert" || result.SequenceLength != 5 || result.ProcessingTimeMs != 1.25 {
		t.Fatalf("lost embedding metadata: %+v, %v", result, err)
	}
	native.Embedding[0] = 0
	if !reflect.DeepEqual(result.Embedding, []float32{3, 4}) {
		t.Fatalf("adapter normalized or aliased the embedding: %v", result.Embedding)
	}
	wantErr := errors.New("execution failed")
	if result, err := nativeEmbeddingResult(nil, wantErr); result != nil || !errors.Is(err, wantErr) {
		t.Fatalf("nil failed native output changed: %+v, %v", result, err)
	}
}
