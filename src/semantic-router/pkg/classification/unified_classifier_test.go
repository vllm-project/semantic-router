package classification

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestUnifiedClassifierNeedsAllThreeRecipeTasks(t *testing.T) {
	if NewUnifiedClassifierFromRecipe(nil) != nil {
		t.Fatal("a missing recipe classifier has no unified classifier")
	}
	if NewUnifiedClassifierFromRecipe(&Classifier{}) != nil {
		t.Fatal("a recipe without intent, PII and security tasks has no unified classifier")
	}
}

func TestUnifiedClassifierRejectsEmptyAndClosedBatches(t *testing.T) {
	classifier := &UnifiedClassifier{classifier: &Classifier{}}
	if !classifier.IsInitialized() {
		t.Fatal("a borrowed recipe classifier serves batches")
	}
	if _, err := classifier.ClassifyBatchContext(context.Background(), nil); err == nil {
		t.Fatal("an empty batch must be rejected")
	}
	if err := classifier.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := classifier.ClassifyBatch([]string{"text"}); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("a closed classifier must refuse batches, got %v", err)
	}
	if classifier.IsInitialized() {
		t.Fatal("a closed classifier is not initialized")
	}
	stats := classifier.GetStats()
	if stats["batch_execution"] != "one_bundle_per_batch" || stats["initialized"] != false {
		t.Fatalf("stats = %v", stats)
	}
}

type partialBatchCategory struct{ failed string }

func (p partialBatchCategory) Classify(ctx context.Context, text string) (tasks.ClassResult, error) {
	result, err := p.ClassifyWithProbabilities(ctx, text)
	return tasks.ClassResult{Class: result.Class, Confidence: result.Confidence}, err
}

func (p partialBatchCategory) ClassifyWithProbabilities(ctx context.Context, text string) (tasks.ClassResultWithProbs, error) {
	if !modelservice.InBundle(ctx) {
		return tasks.ClassResultWithProbs{}, errors.New("missing batch bundle")
	}
	select {
	case <-time.After(time.Duration(3-len(text)) * 5 * time.Millisecond):
	case <-ctx.Done():
		return tasks.ClassResultWithProbs{}, ctx.Err()
	}
	if text == p.failed {
		return tasks.ClassResultWithProbs{}, errors.New("individual failure")
	}
	return tasks.ClassResultWithProbs{Class: len(text) - 1, Confidence: 1, Probabilities: []float32{1, 0, 0}}, nil
}

func partialBatchClassifier(failed string) *UnifiedClassifier {
	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = "prepared-pii"
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.ModelID = "prepared-guard"
	cfg.PromptGuard.Threshold = .5
	return &UnifiedClassifier{classifier: &Classifier{
		Config: cfg, categoryInference: partialBatchCategory{failed: failed}, piiInference: &MockPIIInference{},
		jailbreakInference: &MockJailbreakInference{MockJailbreakInferenceResponse: MockJailbreakInferenceResponse{classifyResult: SequenceClassificationResult{Probabilities: []float32{.99, .01}}}},
		CategoryMapping:    &CategoryMapping{IdxToCategory: map[string]string{"0": "a", "1": "bb", "2": "ccc"}},
		PIIMapping:         &PIIMapping{LabelToIdx: map[string]int{"O": 0}, IdxToLabel: map[string]string{"0": "O"}},
		JailbreakMapping:   &JailbreakMapping{LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}, IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"}},
	}}
}

func TestUnifiedPartialBatchPreservesFirstMiddleLastFailures(t *testing.T) {
	texts := []string{"a", "bb", "ccc"}
	for failedIndex, failed := range texts {
		classifier := partialBatchClassifier(failed)
		result, err := classifier.ClassifyBatchPartialContext(context.Background(), texts)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.ItemErrors) != len(texts) || result.BatchSize != len(texts) {
			t.Fatalf("wrong cardinality: %+v", result)
		}
		for index, text := range texts {
			if index == failedIndex {
				if result.ItemErrors[index] == nil {
					t.Fatalf("missing error at %d", index)
				}
				continue
			}
			if result.ItemErrors[index] != nil || result.IntentResults[index].Category != text {
				t.Fatalf("sibling %d: %+v %v", index, result.IntentResults[index], result.ItemErrors[index])
			}
		}
		if result, err := classifier.ClassifyBatchContext(context.Background(), texts); err == nil || result != nil {
			t.Fatal("compatibility method lost fail-fast semantics")
		}
	}
}

func TestUnifiedPartialBatchRejectsBlankAndCanceledItems(t *testing.T) {
	classifier := partialBatchClassifier("")
	result, err := classifier.ClassifyBatchPartialContext(context.Background(), []string{"a", " "})
	if err != nil || result.ItemErrors[0] != nil || !errors.Is(result.ItemErrors[1], binding.ErrInvalidInput) {
		t.Fatalf("blank item: %+v %v", result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err = classifier.ClassifyBatchPartialContext(ctx, []string{"a", "bb"})
	if err != nil {
		t.Fatal(err)
	}
	for index, itemErr := range result.ItemErrors {
		if !errors.Is(itemErr, context.Canceled) {
			t.Fatalf("item %d: %v", index, itemErr)
		}
	}
}
