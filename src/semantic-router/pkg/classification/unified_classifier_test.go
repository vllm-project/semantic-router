package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
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
