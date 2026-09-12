//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// This test requires actual AMD hardware and an ORT build containing MIGraphX.
// It asserts GPU node execution, rather than inferring it from a device listing.
func TestMIGraphXExecutesWithoutCPUFallback(t *testing.T) {
	if os.Getenv("ORT_TEST_MIGRAPHX") != "1" {
		t.Skip("set ORT_TEST_MIGRAPHX=1 on an AMD GPU host")
	}
	options := fixture("sequence")
	options.Provider = "migraphx"
	options.ProfilePrefix = filepath.Join(t.TempDir(), "migraphx-profile")
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if _, inferErr := model.Classify("hello world"); inferErr != nil {
		t.Fatal(inferErr)
	}
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.CompletedInferences != 1 || len(info.Sessions) != 1 || !info.Sessions[0].CPUFallbackDisabled {
		t.Fatalf("missing strict GPU execution evidence: %+v", info)
	}
	paths, err := model.FinishProfiling()
	if err != nil {
		t.Fatal(err)
	}
	var gpuNodes int
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var events []struct {
			Args struct {
				Provider string `json:"provider"`
			} `json:"args"`
		}
		if err := json.Unmarshal(data, &events); err != nil {
			t.Fatal(err)
		}
		for _, event := range events {
			if event.Args.Provider == "CPUExecutionProvider" {
				t.Fatal("CPU node executed despite a strict GPU request")
			}
			if event.Args.Provider == "MIGraphXExecutionProvider" {
				gpuNodes++
			}
		}
	}
	if gpuNodes == 0 {
		t.Fatal("profile contains no MIGraphX node execution")
	}
	t.Logf("MIGraphX executed %d profiled nodes; CPU node count = 0", gpuNodes)
}

func fixture(kind string) Options {
	return Options{ModelPath: filepath.Join("testdata", kind), Provider: "cpu", IntraThreads: 1}
}

func errorKind(t *testing.T, err error, kind string) {
	t.Helper()
	var providerError *Error
	if !errors.As(err, &providerError) || providerError.Kind != kind {
		t.Fatalf("expected %s error, got %v", kind, err)
	}
}

func TestIndependentSequencesAndSharedOwnership(t *testing.T) {
	first, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	shared, err := first.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer shared.Close()
	before, err := first.Classify("hello")
	if err != nil {
		t.Fatal(err)
	}
	after, err := second.Classify("world")
	if err != nil {
		t.Fatal(err)
	}
	if before.Label != "positive" || len(before.Probabilities) != 2 || before.Confidence >= after.Confidence {
		t.Fatalf("real graph must respond to different token IDs: %+v %+v", before, after)
	}
	if closeErr := first.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if _, closedErr := first.Classify("hello"); closedErr == nil {
		t.Fatal("closed owner accepted inference")
	} else {
		errorKind(t, closedErr, "closed")
	}
	for _, live := range []*SequenceClassifier{second, shared} {
		if _, inferErr := live.Classify("hello"); inferErr != nil {
			t.Fatalf("closing another owner unloaded live session: %v", inferErr)
		}
	}
	info, err := shared.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.CompletedInferences != 2 || info.TaskLimit != 512 || info.ModelLimit != 4096 || info.Sessions[0].Provider != "CPUExecutionProvider" {
		t.Fatalf("incorrect task or execution evidence: %+v", info)
	}
}

func TestClassificationBudgetAndUTF8TokenOffsets(t *testing.T) {
	options := fixture("sequence")
	options.MaxInputTokens = 2
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	_, err = model.Classify("hello world test")
	errorKind(t, err, "input_limit")
	options.Overflow = "truncate_right"
	truncated, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer truncated.Close()
	result, err := truncated.Classify("hello world test")
	if err != nil {
		t.Fatal(err)
	}
	if !result.Input.Truncated || result.Input.OriginalTokens != 3 || result.Input.ProcessedTokens != 2 {
		t.Fatalf("truncation was not reported: %+v", result.Input)
	}
	tokens, err := LoadTokenClassifier(fixture("token"))
	if err != nil {
		t.Fatal(err)
	}
	defer tokens.Close()
	text := "秘密 hello"
	spans, err := tokens.Detect(text)
	if err != nil {
		t.Fatal(err)
	}
	if spans.OffsetUnit != "utf8_bytes" || len(spans.Spans) != 2 || spans.Spans[0].End != len("秘密") {
		t.Fatalf("unexpected token spans: %+v", spans)
	}
	for _, span := range spans.Spans {
		if text[span.Start:span.End] != span.Text {
			t.Fatalf("span does not reference original input: %+v", span)
		}
	}
}

func assertEmbedding(t *testing.T, result EmbeddingResult, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	var norm float64
	for _, value := range result.Values {
		norm += float64(value * value)
	}
	if len(result.Values) != 2 || math.Abs(norm-1) > 1e-5 {
		t.Fatalf("invalid normalized embedding: %+v", result)
	}
}

func TestTextAndMultimodalInstancePaths(t *testing.T) {
	embedding, err := LoadEmbeddingModel(fixture("embedding"))
	if err != nil {
		t.Fatal(err)
	}
	defer embedding.Close()
	result, err := embedding.Encode("hello world", 0, 2)
	assertEmbedding(t, result, err)
	multi, err := LoadMultiModal(fixture("multimodal"))
	if err != nil {
		t.Fatal(err)
	}
	defer multi.Close()
	clone, err := multi.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if closeErr := multi.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	result, err = clone.EncodeText("hello", 2)
	assertEmbedding(t, result, err)
	result, err = clone.EncodeImage(make([]float32, 12), 2, 2, 2)
	assertEmbedding(t, result, err)
	result, err = clone.EncodeAudio(make([]float32, 20), 2, 10, 2)
	assertEmbedding(t, result, err)
}

func TestConcurrentCloseDoesNotUnloadSharedNativeCalls(t *testing.T) {
	model, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	var workers sync.WaitGroup
	for i := 0; i < 8; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for n := 0; n < 30; n++ {
				if _, err := clone.Classify("hello world"); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	workers.Wait()
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestProfileRecordsRealExecution(t *testing.T) {
	options := fixture("sequence")
	options.ProfilePrefix = filepath.Join(t.TempDir(), "ort-profile")
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if _, inferErr := model.Classify("hello"); inferErr != nil {
		t.Fatal(inferErr)
	}
	paths, err := model.FinishProfiling()
	if err != nil || len(paths) != 1 {
		t.Fatalf("profile paths %v: %v", paths, err)
	}
	profile, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(profile), "CPUExecutionProvider") || !strings.Contains(string(profile), "kernel_time") {
		t.Fatal("profile does not prove real CPU node execution")
	}
}

func TestPreparationFailurePreservesExistingInstance(t *testing.T) {
	model, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	bad := fixture("sequence")
	bad.ModelFile = "missing.onnx"
	if _, err := LoadSequenceClassifier(bad); err == nil {
		t.Fatal("invalid candidate loaded")
	}
	if _, inferErr := model.Classify("hello"); inferErr != nil {
		t.Fatal(inferErr)
	}
	bad = fixture("sequence")
	bad.MaxInputTokens = 513
	if _, err := LoadSequenceClassifier(bad); err == nil {
		t.Fatal("classification budget exceeded task's 512-token limit")
	}
}

func TestOwnedEmbeddingWindowsUseEffectiveBudgetAndSurvivePeerClose(t *testing.T) {
	options := fixture("embedding")
	options.MaxInputTokens = 2
	model, err := LoadEmbeddingModel(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	peer, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer peer.Close()
	text := "hello 秘密 world hello 秘密"
	windows, err := model.Windows(text, 0)
	if err != nil || len(windows) != 3 {
		t.Fatalf("effective tokenizer windows: %+v %v", windows, err)
	}
	info, err := model.Info()
	if err != nil || info.CompletedInferences != 0 {
		t.Fatalf("windowing must not pretend to execute a model: %+v %v", info, err)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	for _, window := range windows {
		if window.Start < 0 || window.End > len(text) || window.Start >= window.End {
			t.Fatalf("invalid UTF-8 window: %+v", window)
		}
		result, encodeErr := peer.Encode(text[window.Start:window.End], 0, 2)
		if encodeErr != nil || result.Input == nil || result.Input.Truncated || result.Input.ProcessedTokens > 2 {
			t.Fatalf("window did not fit actual model tokenizer: %+v %v", result, encodeErr)
		}
	}
	clamped, err := peer.Windows(text, 999)
	if err != nil || len(clamped) != len(windows) {
		t.Fatalf("caller expanded the effective model budget: %+v %v", clamped, err)
	}
	_, err = model.Windows(text, 0)
	errorKind(t, err, "closed")
	_, err = peer.Windows(text, -1)
	errorKind(t, err, "invalid_input")
}
