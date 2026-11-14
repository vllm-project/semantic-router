package classification

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
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

type partialBatchDecider struct{ failed string }

func (p partialBatchDecider) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	if !modelservice.InBundle(ctx) {
		return modelservice.Response{}, errors.New("missing batch bundle")
	}
	select {
	case <-time.After(time.Duration(3-len(request.State)) * 5 * time.Millisecond):
	case <-ctx.Done():
		return modelservice.Response{}, ctx.Err()
	}
	if request.State == p.failed {
		return modelservice.Response{}, errors.New("individual failure")
	}
	answers := make(map[string]modelservice.Answer)
	for _, question := range request.Questions {
		answers[question.ID] = modelservice.Answer{Type: "choice", Choice: request.State, Probabilities: map[string]float64{request.State: 1, "benign": 1, "jailbreak": 0}, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

func partialBatchClassifier(failed string) *UnifiedClassifier {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.SetDecisionDecider(partialBatchDecider{failed: failed})
	return &UnifiedClassifier{classifier: classifier}
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

type selectiveBatchDecider struct {
	mu    sync.Mutex
	asked []string
}

func (d *selectiveBatchDecider) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	answers := make(map[string]modelservice.Answer)
	for _, question := range request.Questions {
		d.asked = append(d.asked, question.ID)
		if question.ID == "preset:domain" {
			return modelservice.Response{}, errors.New("domain unavailable")
		}
		answers[question.ID] = modelservice.Answer{Type: "span", InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

func TestUnifiedPartialBatchSelectedTaskKeepsLegacyQuestionsAndIgnoresOtherErrors(t *testing.T) {
	d := &selectiveBatchDecider{}
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.SetDecisionDecider(d)
	unified := &UnifiedClassifier{classifier: classifier}
	result, err := unified.ClassifyBatchPartialTaskContext(context.Background(), []string{"one", "two"}, "pii")
	if err != nil || result.ItemErrors[0] != nil || result.ItemErrors[1] != nil {
		t.Fatalf("PII results lost to another task: %+v %v", result, err)
	}
	counts := make(map[string]int)
	for _, id := range d.asked {
		counts[id]++
	}
	for _, id := range []string{"preset:domain", "preset:pii", "preset:jailbreak"} {
		if counts[id] != 2 {
			t.Fatalf("legacy question set not asked once per item: %v", counts)
		}
	}
	if len(d.asked) != 6 {
		t.Fatalf("unexpected logical questions: %v", d.asked)
	}
}

type taskErrorDecider struct{}

func (taskErrorDecider) Decide(_ context.Context, _ string, req modelservice.Request) (modelservice.Response, error) {
	if len(req.Questions) == 0 {
		return modelservice.Response{}, errors.New("missing question")
	}
	switch req.Questions[0].ID {
	case "preset:domain":
		return modelservice.Response{}, admission.ErrQueueFull
	case "preset:pii":
		return modelservice.Response{}, binding.ErrInputLimit
	default:
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
}

func TestUnifiedLegacyBatchKeepsFirstTaskErrorOrder(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.SetDecisionDecider(taskErrorDecider{})
	unified := &UnifiedClassifier{classifier: classifier}
	_, err := unified.ClassifyBatchContext(context.Background(), []string{"text"})
	if !errors.Is(err, admission.ErrQueueFull) || errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("legacy task-order error changed: %v", err)
	}
	if unified.stats.TotalBatches != 0 || unified.stats.TotalTexts != 0 {
		t.Fatalf("failed legacy batch changed statistics: %+v", unified.stats)
	}
	partial, err := unified.ClassifyBatchPartialContext(context.Background(), []string{"text"})
	if err != nil || !errors.Is(partial.ItemErrors[0], admission.ErrQueueFull) || !errors.Is(partial.ItemErrors[0], binding.ErrInputLimit) {
		t.Fatalf("typed partial item lost failures: %+v %v", partial, err)
	}
	for _, tc := range []struct {
		task string
		want error
	}{
		{"pii", binding.ErrInputLimit},
		{"security", modelservice.ErrUnavailable},
	} {
		selected, err := unified.ClassifyBatchPartialTaskContext(context.Background(), []string{"text"}, tc.task)
		if err != nil || !errors.Is(selected.ItemErrors[0], tc.want) || errors.Is(selected.ItemErrors[0], admission.ErrQueueFull) {
			t.Fatalf("%s reported unrelated task errors: %+v %v", tc.task, selected, err)
		}
	}
}

type failedExchangeDecider struct{}

func (failedExchangeDecider) Decide(context.Context, string, modelservice.Request) (modelservice.Response, error) {
	return modelservice.Response{}, modelservice.ErrUnavailable
}

func TestUnifiedPartialBatchSelectedTaskReportsSharedExchangeFailure(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.SetDecisionDecider(failedExchangeDecider{})
	unified := &UnifiedClassifier{classifier: classifier}
	for _, selected := range []string{"pii", "security", "all"} {
		result, err := unified.ClassifyBatchPartialTaskContext(context.Background(), []string{"one", "two"}, selected)
		if err != nil {
			t.Fatal(err)
		}
		for i, itemErr := range result.ItemErrors {
			if !errors.Is(itemErr, modelservice.ErrUnavailable) {
				t.Fatalf("%s item %d: %v", selected, i, itemErr)
			}
		}
	}
}
