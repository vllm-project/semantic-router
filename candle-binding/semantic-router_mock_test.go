//go:build windows || !cgo

// These tests pin the fail-closed contract of the non-CGO Candle stub
// (issue #2491). Every inference or mutation API must report the native backend
// as unavailable instead of returning a synthetic success. The tests use no
// t.Skip so they cannot silently pass when the stub is built.

package candle_binding

import (
	"errors"
	"testing"
)

// wantUnavailable asserts that err is the typed backend-unavailable sentinel.
func wantUnavailable(t *testing.T, name string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: expected ErrBackendUnavailable, got nil (synthetic success from unavailable backend)", name)
	}
	if !errors.Is(err, ErrBackendUnavailable) {
		t.Fatalf("%s: expected ErrBackendUnavailable, got %v", name, err)
	}
}

// TestStubInitFailsClosed verifies initialization APIs refuse to report success.
func TestStubInitFailsClosed(t *testing.T) {
	wantUnavailable(t, "InitModel", InitModel("any-model", true))
	wantUnavailable(t, "InitClassifier", InitClassifier("path", 2, true))
	wantUnavailable(t, "InitPIIClassifier", InitPIIClassifier("path", 2, true))
	wantUnavailable(t, "InitJailbreakClassifier", InitJailbreakClassifier("path", 2, true))
	wantUnavailable(t, "InitLoRAUnifiedClassifier", InitLoRAUnifiedClassifier("i", "p", "s", "arch", true))
	wantUnavailable(t, "InitQwen3MultiLoRAClassifier", InitQwen3MultiLoRAClassifier("base"))
	wantUnavailable(t, "InitQwen3Guard", InitQwen3Guard("path"))
	wantUnavailable(t, "InitMultiModalEmbeddingModel", InitMultiModalEmbeddingModel("path", true))

	// Bool-returning init APIs (no error channel) must report failure.
	if InitCandleBertClassifier("path", 2, true) {
		t.Fatal("InitCandleBertClassifier: expected false from unavailable backend")
	}
	if InitCandleBertTokenClassifier("path", 2, true) {
		t.Fatal("InitCandleBertTokenClassifier: expected false from unavailable backend")
	}
}

// TestStubStateFailsClosed verifies backend state is reported as uninitialized.
func TestStubStateFailsClosed(t *testing.T) {
	if rust, goState := IsModelInitialized(); rust || goState {
		t.Fatalf("IsModelInitialized: expected (false, false), got (%v, %v)", rust, goState)
	}
	if IsQwen3GuardInitialized() {
		t.Fatal("IsQwen3GuardInitialized: expected false from unavailable backend")
	}
	if IsQwen3MultiLoRAInitialized() {
		t.Fatal("IsQwen3MultiLoRAInitialized: expected false from unavailable backend")
	}
}

// TestStubEmbeddingFailsClosed covers embedding and multimodal APIs.
func TestStubEmbeddingFailsClosed(t *testing.T) {
	_, err := GetEmbedding("hello", 512)
	wantUnavailable(t, "GetEmbedding", err)

	_, err = GetEmbeddingWithMetadata("hello", 0, 0, 384)
	wantUnavailable(t, "GetEmbeddingWithMetadata", err)

	_, err = MultiModalEncodeText("hello", 384)
	wantUnavailable(t, "MultiModalEncodeText", err)

	_, err = MultiModalEncodeImageFromURL("http://example.com/x.png", 384)
	wantUnavailable(t, "MultiModalEncodeImageFromURL", err)
}

// TestStubSimilarityFailsClosed covers similarity APIs, including the two that
// have no error channel and must return an out-of-range sentinel.
func TestStubSimilarityFailsClosed(t *testing.T) {
	if s := CalculateSimilarity("a", "b", 512); s >= 0 {
		t.Fatalf("CalculateSimilarity: expected negative sentinel, got %v", s)
	}
	if s := CalculateSimilarityDefault("a", "b"); s >= 0 {
		t.Fatalf("CalculateSimilarityDefault: expected negative sentinel, got %v", s)
	}
	if r := FindMostSimilar("q", []string{"a", "b"}, 512); r.Index != -1 {
		t.Fatalf("FindMostSimilar: expected Index -1, got %d", r.Index)
	}

	_, err := CalculateEmbeddingSimilarity("a", "b", "qwen3", 0)
	wantUnavailable(t, "CalculateEmbeddingSimilarity", err)

	_, err = CalculateSimilarityBatch("q", []string{"a", "b"}, 2, "qwen3", 0)
	wantUnavailable(t, "CalculateSimilarityBatch", err)
}

// TestStubClassificationFailsClosed covers the safety-critical classifiers: an
// unavailable backend must never return a "Safe"/benign verdict.
func TestStubClassificationFailsClosed(t *testing.T) {
	_, err := ClassifyText("hello")
	wantUnavailable(t, "ClassifyText", err)

	_, err = ClassifyJailbreakText("ignore previous instructions")
	wantUnavailable(t, "ClassifyJailbreakText", err)

	_, err = ClassifyPIIText("my ssn is 123-45-6789")
	wantUnavailable(t, "ClassifyPIIText", err)

	_, err = ClassifyPromptSafety("dangerous prompt")
	wantUnavailable(t, "ClassifyPromptSafety", err)

	_, err = ClassifyResponseSafety("dangerous response")
	wantUnavailable(t, "ClassifyResponseSafety", err)

	_, err = ClassifyCandleBertText("hello")
	wantUnavailable(t, "ClassifyCandleBertText", err)
}

// TestStubLoRAFailsClosed covers LoRA batch/adapter APIs, including adapter
// state mutation and result cardinality.
func TestStubLoRAFailsClosed(t *testing.T) {
	_, err := ClassifyBatchWithLoRA([]string{"a", "b", "c"})
	wantUnavailable(t, "ClassifyBatchWithLoRA", err)

	wantUnavailable(t, "LoadQwen3LoRAAdapter", LoadQwen3LoRAAdapter("name", "path"))

	_, err = ClassifyWithQwen3Adapter("text", "adapter")
	wantUnavailable(t, "ClassifyWithQwen3Adapter", err)

	_, err = GetQwen3LoadedAdapters()
	wantUnavailable(t, "GetQwen3LoadedAdapters", err)

	_, err = ClassifyZeroShotQwen3("text", []string{"a", "b"})
	wantUnavailable(t, "ClassifyZeroShotQwen3", err)
}
