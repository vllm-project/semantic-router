//go:build !windows && cgo

package openvino_binding

import (
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

const ownedText = "alpha beta gamma"

func ownedOptions(t *testing.T, name string, maximum int, overflow string) ModelOptions {
	t.Helper()
	directory := os.Getenv("OPENVINO_OWNED_FIXTURE_DIR")
	if directory == "" {
		t.Skip("set OPENVINO_OWNED_FIXTURE_DIR after running scripts/create_owned_fixture.py")
	}
	path := filepath.Join(directory, name, "openvino_model.xml")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("required owned fixture %s: %v", path, err)
	}
	return ModelOptions{
		ModelPath: path, Device: "CPU", MaxTokens: maximum,
		Overflow: overflow, EndTokenIDs: []int{2}, PadTokenID: 0,
	}
}

func loadOwnedPair(t *testing.T, variant string, maximum int, overflow string) (*EmbeddingModel, *ClassifierModel) {
	t.Helper()
	embedding, err := LoadEmbeddingModel(ownedOptions(t, "embedding_"+variant, maximum, overflow))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embedding.Close() })
	classifier, err := LoadClassifierModel(ownedOptions(t, "classifier_"+variant, maximum, overflow), 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	return embedding, classifier
}

func ownedExpected(variant string, ids []int) ([]float32, []float32) {
	total, count := float64(0), float64(0)
	for _, id := range ids {
		if id != 0 {
			total += float64(id)
			count++
		}
	}
	mean := total / count
	embedding := []float32{float32(mean), float32(2*mean + 1), float32(3*mean + 2)}
	// Independently compute the fixture's exact binary-fraction logits before
	// the binding's float32 softmax; CPU plugins may reduce model precision.
	logits := []float64{total / 64, count / 8, 1}
	if variant == "b" {
		embedding = []float32{float32(11 - mean), float32(0.5*mean - 7), float32(2*mean + 5)}
		logits = []float64{1, -total / 32, count / 16}
	}
	denominator := float64(0)
	for i := range logits {
		logits[i] = math.Exp(logits[i])
		denominator += logits[i]
	}
	probabilities := make([]float32, len(logits))
	for i := range logits {
		probabilities[i] = float32(logits[i] / denominator)
	}
	return embedding, probabilities
}

func ownedVectorError(actual, expected []float32) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("length %d, want %d", len(actual), len(expected))
	}
	for i, value := range actual {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-expected[i])) > 1e-5 {
			return fmt.Errorf("value[%d]=%g, want %g", i, value, expected[i])
		}
	}
	return nil
}

func assertOwnedPair(t *testing.T, embedding *EmbeddingModel, classifier *ClassifierModel, variant, text string, ids []int, usage InputUsage) {
	t.Helper()
	wantEmbedding, wantProbabilities := ownedExpected(variant, ids)
	vector, err := embedding.Embed(text)
	if err != nil {
		t.Fatal(err)
	}
	if vectorErr := ownedVectorError(vector.Values, wantEmbedding); vectorErr != nil {
		t.Fatal(vectorErr)
	}
	result, err := classifier.Classify(text)
	if err != nil {
		t.Fatal(err)
	}
	if vectorErr := ownedVectorError(result.Probabilities, wantProbabilities); vectorErr != nil {
		t.Fatal(vectorErr)
	}
	wantClass := 0
	for index, probability := range wantProbabilities {
		if probability > wantProbabilities[wantClass] {
			wantClass = index
		}
	}
	if result.Class != wantClass || result.Confidence != result.Probabilities[result.Class] {
		t.Fatalf("classification result: %+v", result)
	}
	if vector.Input != usage || result.Input != usage {
		t.Fatalf("input usage: embedding=%+v classifier=%+v, want %+v", vector.Input, result.Input, usage)
	}
}

func TestOwnedHandlesRemainIndependent(t *testing.T) {
	a, ac := loadOwnedPair(t, "a", 16, "reject")
	b, bc := loadOwnedPair(t, "b", 16, "reject")
	usage := InputUsage{OriginalTokens: 5, ProcessedTokens: 5}
	for range 3 {
		assertOwnedPair(t, a, ac, "a", ownedText, []int{1, 4, 5, 6, 2}, usage)
		assertOwnedPair(t, b, bc, "b", ownedText, []int{1, 14, 15, 16, 2}, usage)
	}
	_ = a.Close()
	_ = a.Close()
	if _, err := a.Embed(ownedText); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed embedding: %v", err)
	}
	if _, err := ac.Classify(ownedText); err != nil {
		t.Fatalf("closing embedding affected classifier: %v", err)
	}
	_ = ac.Close()
	_ = ac.Close()
	if _, err := ac.Classify(ownedText); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed classifier: %v", err)
	}
	assertOwnedPair(t, b, bc, "b", ownedText, []int{1, 14, 15, 16, 2}, usage)
}

func TestOwnedBudgetsAndPadding(t *testing.T) {
	t.Run("padding", func(t *testing.T) {
		e, c := loadOwnedPair(t, "a", 16, "reject")
		assertOwnedPair(t, e, c, "a", "alpha [PAD] beta", []int{1, 4, 0, 5, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
	})
	t.Run("exact", func(t *testing.T) {
		e, c := loadOwnedPair(t, "a", 5, "reject")
		assertOwnedPair(t, e, c, "a", ownedText, []int{1, 4, 5, 6, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
	})
	t.Run("truncate_preserves_sep", func(t *testing.T) {
		e, c := loadOwnedPair(t, "a", 4, "truncate")
		assertOwnedPair(t, e, c, "a", ownedText, []int{1, 4, 5, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 4, Truncated: true})
	})
	t.Run("reject", func(t *testing.T) {
		e, c := loadOwnedPair(t, "a", 4, "reject")
		if result, err := e.Embed(ownedText); !errors.Is(err, ErrInputTooLong) || len(result.Values) != 0 {
			t.Fatalf("embedding over budget: %+v %v", result, err)
		}
		if result, err := c.Classify(ownedText); !errors.Is(err, ErrInputTooLong) || len(result.Probabilities) != 0 {
			t.Fatalf("classifier over budget: %+v %v", result, err)
		}
	})
	t.Run("envelope_cannot_fit", func(t *testing.T) {
		e, c := loadOwnedPair(t, "a", 1, "truncate")
		if _, err := e.Embed(ownedText); err == nil {
			t.Fatal("accepted a one-token budget for CLS + SEP")
		}
		if _, err := c.Classify(ownedText); err == nil {
			t.Fatal("classifier accepted a one-token budget for CLS + SEP")
		}
	})
}

func TestOwnedCountsBeyondDeclaredModelLimit(t *testing.T) {
	// The fixture declares 64 positions, but its tokenizer must expose all 98
	// tokens before the binding applies its own lower request budget.
	text := strings.TrimSpace(strings.Repeat("alpha ", 96))
	e, c := loadOwnedPair(t, "a", 16, "truncate")
	ids := make([]int, 16)
	ids[0], ids[15] = 1, 2
	for index := 1; index < 15; index++ {
		ids[index] = 4
	}
	assertOwnedPair(t, e, c, "a", text, ids, InputUsage{OriginalTokens: 98, ProcessedTokens: 16, Truncated: true})
	reject, rejectClassifier := loadOwnedPair(t, "a", 64, "reject")
	if _, err := reject.Embed(text); !errors.Is(err, ErrInputTooLong) {
		t.Fatalf("embedding hid tokenizer overflow: %v", err)
	}
	if _, err := rejectClassifier.Classify(text); !errors.Is(err, ErrInputTooLong) {
		t.Fatalf("classifier hid tokenizer overflow: %v", err)
	}
}

func TestOwnedRejectsNullText(t *testing.T) {
	e, c := loadOwnedPair(t, "a", 16, "reject")
	if _, err := e.Embed("alpha\x00beta"); err == nil {
		t.Fatal("embedding silently accepted a null byte")
	}
	if _, err := c.Classify("alpha\x00beta"); err == nil {
		t.Fatal("classifier silently accepted a null byte")
	}
	assertOwnedPair(t, e, c, "a", ownedText, []int{1, 4, 5, 6, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
}

func TestOwnedInitializationCanRetry(t *testing.T) {
	options := ownedOptions(t, "embedding_a", 16, "reject")
	if os.Getenv("OPENVINO_OWNED_RETRY_CHILD") != "1" {
		// #nosec G204 -- only re-executes this test binary with fixed test-selection arguments.
		command := exec.Command(os.Args[0], "-test.run=^TestOwnedInitializationCanRetry$", "-test.v")
		command.Env = append(os.Environ(), "OPENVINO_OWNED_RETRY_CHILD=1")
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("first-core retry subprocess: %v\n%s", err, output)
		}
		return
	}
	extension := os.Getenv("OPENVINO_TOKENIZERS_LIB")
	if extension == "" {
		t.Fatal("OPENVINO_TOKENIZERS_LIB must be set for the real fixture")
	}
	t.Setenv("OPENVINO_TOKENIZERS_LIB", filepath.Join(t.TempDir(), "missing.so"))
	if model, err := LoadEmbeddingModel(options); err == nil || model != nil {
		t.Fatal("accepted a missing tokenizer extension")
	}
	t.Setenv("OPENVINO_TOKENIZERS_LIB", extension)
	e, c := loadOwnedPair(t, "a", 16, "reject")
	assertOwnedPair(t, e, c, "a", ownedText, []int{1, 4, 5, 6, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
	bad := ownedOptions(t, "classifier_a", 16, "reject")
	bad.ModelPath = filepath.Join(t.TempDir(), "missing.xml")
	if model, err := LoadClassifierModel(bad, 3); err == nil || model != nil {
		t.Fatal("accepted a missing classifier")
	}
	_, retry := loadOwnedPair(t, "b", 16, "reject")
	if _, err := retry.Classify(ownedText); err != nil {
		t.Fatalf("classifier could not recover after initialization failure: %v", err)
	}
}

func TestOwnedReopenOnSameThread(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	addresses := map[string]bool{}
	reused := 0
	for iteration := range 32 {
		variant, offset := "a", 4
		if iteration%2 == 1 {
			variant, offset = "b", 14
		}
		e, c := loadOwnedPair(t, variant, 16, "reject")
		address := fmt.Sprintf("%p", e.handle)
		if addresses[address] {
			reused++
		}
		addresses[address] = true
		assertOwnedPair(t, e, c, variant, ownedText, []int{1, offset, offset + 1, offset + 2, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
		_ = e.Close()
		_ = c.Close()
	}
	// Reuse is allocator-dependent; correctness must not depend on its occurrence.
	t.Logf("32 same-thread reopen cycles; observed %d reused embedding handle addresses", reused)
}

func TestOwnedConcurrentInferAndClose(t *testing.T) {
	e, c := loadOwnedPair(t, "a", 16, "reject")
	peer, peerClassifier := loadOwnedPair(t, "b", 16, "reject")
	wantEmbedding, wantProbabilities := ownedExpected("a", []int{1, 4, 5, 6, 2})
	const workers = 6
	ready := make(chan struct{}, workers)
	errorsSeen := make(chan error, workers)
	var group sync.WaitGroup
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			for iteration := range 30 {
				vector, embedErr := e.Embed(ownedText)
				result, classifyErr := c.Classify(ownedText)
				if iteration == 0 {
					ready <- struct{}{}
				}
				if embedErr == nil {
					embedErr = ownedVectorError(vector.Values, wantEmbedding)
				}
				if classifyErr == nil {
					classifyErr = ownedVectorError(result.Probabilities, wantProbabilities)
				}
				for _, err := range []error{embedErr, classifyErr} {
					if err != nil && !errors.Is(err, ErrClosed) {
						errorsSeen <- err
						return
					}
				}
			}
		}()
	}
	for range workers {
		<-ready
	}
	_ = e.Close()
	_ = c.Close()
	group.Wait()
	close(errorsSeen)
	for err := range errorsSeen {
		t.Error(err)
	}
	assertOwnedPair(t, peer, peerClassifier, "b", ownedText, []int{1, 14, 15, 16, 2}, InputUsage{OriginalTokens: 5, ProcessedTokens: 5})
	if _, err := e.Embed(ownedText); !errors.Is(err, ErrClosed) {
		t.Fatalf("embedding after concurrent close: %v", err)
	}
	if _, err := c.Classify(ownedText); !errors.Is(err, ErrClosed) {
		t.Fatalf("classifier after concurrent close: %v", err)
	}
}
