//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestOwnedTypedScoresAndWindowsUseActualHeadBudget(t *testing.T) {
	base := nativeHeadlessFixturePart(t, 0, true)
	categorical := nativeHeadlessFixturePart(t, 1, false)
	independent := nativeHeadlessFixturePart(t, 1, false)
	for _, path := range []string{base, categorical, independent} {
		raw, err := os.ReadFile(filepath.Join(path, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		var metadata map[string]any
		if decodeErr := json.Unmarshal(raw, &metadata); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		metadata["max_position_embeddings"] = 2048
		if path == independent {
			metadata["problem_type"] = "multi_label_classification"
		}
		raw, err = json.Marshal(metadata)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "config.json"), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	runtime := New(nil)
	ctx := context.Background()
	spec := config.ResolvedModelBinding{Recipe: "one", Name: "binary", Binding: config.ModelBinding{Deployment: "encoder", Adapter: "modernbert", Head: categorical, Contract: config.RemoteClassifierContractLabelDistribution}, Deployment: config.ModelDeployment{Artifact: base, Provider: "candle", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 1024, Overflow: "reject"}}}
	sequence, err := runtime.Sequence(ctx, spec)
	if err != nil {
		t.Fatal(err)
	}
	defer sequence.Close()
	scoreSpec := spec
	scoreSpec.Name = "hazard"
	scoreSpec.Binding.Contract = config.RemoteClassifierContractLabelScores
	scoreSpec.Binding.Head = independent
	scores, err := runtime.Scores(ctx, scoreSpec)
	if err != nil {
		t.Fatal(err)
	}
	defer scores.Close()
	if scores.Capability().Limits.EffectiveTokens() != 1024 {
		t.Fatalf("lost admitted budget: %+v", scores.Capability())
	}
	out, err := scores.Call(ctx, "one", "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if out.Scores[0]+out.Scores[1] <= 1 {
		t.Fatal("independent head was normalized")
	}
	if _, err = scores.Call(ctx, "other", "hello"); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe admitted: %v", err)
	}
	if err = sequence.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = scores.Call(ctx, "one", "hello"); err != nil {
		t.Fatal("closing sibling removed score head", err)
	}
	if _, err = scores.Call(ctx, "one", strings.Repeat("hello ", 1025)); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("whole input budget not enforced: %v", err)
	}
	scoreSpec.Deployment.Input.MaxTokens = 4096
	window := tasks.TextWindowsRequest{Text: strings.Repeat("hello ", 4096), Size: 128, Overlap: 16}
	scoreWindows, err := runtime.ScoreWindows(ctx, scoreSpec, window)
	if err != nil {
		t.Fatal(err)
	}
	defer scoreWindows.Close()
	windows, err := scoreWindows.Call(ctx, "one", window)
	if err != nil {
		t.Fatal(err)
	}
	if windows.ContentTokens != 4096 || windows.Input.ProcessedTokens != 4096 || len(windows.Windows) < 2 || scoreWindows.Capability().Limits.Overflow != "window" {
		t.Fatalf("window coverage/usage lost: %+v", windows)
	}
	changed := window
	changed.Size = 64
	if _, err = scoreWindows.Call(ctx, "one", changed); !errors.Is(err, binding.ErrInvalidInput) {
		t.Fatalf("prepared window policy changed: %v", err)
	}
	spec.Deployment.Input.MaxTokens = 4096
	categoricalWindows, err := runtime.SequenceWindows(ctx, spec, window)
	if err != nil {
		t.Fatal(err)
	}
	defer categoricalWindows.Close()
	if _, err := categoricalWindows.Call(ctx, "one", window); err != nil {
		t.Fatal(err)
	}
	invalid := scoreSpec
	invalid.Deployment.Input.MaxTokens = 1024
	invalid.Binding.Contract = config.RemoteClassifierContractLabelDistribution
	if head, err := runtime.Sequence(ctx, invalid); err == nil {
		head.Close()
		t.Fatal("multi-label head bound as categorical")
	}
	invalid = scoreSpec
	invalid.Deployment.Input.MaxTokens = 2049
	if head, err := runtime.Scores(ctx, invalid); err == nil {
		head.Close()
		t.Fatal("budget beyond actual checkpoint accepted")
	}
}

func TestOwnedTypedORTScoresAndWindowsUseActualSession(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires the real ONNX Runtime library")
	}
	directory := t.TempDir()
	fixture := filepath.Join("..", "..", "..", "..", "..", "onnx-binding", "instance", "testdata", "sequence")
	for _, name := range []string{"config.json", "tokenizer.json", "model.onnx"} {
		data, readErr := os.ReadFile(filepath.Join(fixture, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if name == "config.json" {
			var metadata map[string]any
			if decodeErr := json.Unmarshal(data, &metadata); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			metadata["problem_type"] = "multi_label_classification"
			metadata["max_position_embeddings"] = 32768
			data, readErr = json.Marshal(metadata)
			if readErr != nil {
				t.Fatal(readErr)
			}
		}
		if writeErr := os.WriteFile(filepath.Join(directory, name), data, 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	runtime := New(nil)
	ctx := context.Background()
	spec := config.ResolvedModelBinding{Recipe: "one", Name: "hazard", Binding: config.ModelBinding{Deployment: "head", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelScores}, Deployment: config.ModelDeployment{Provider: "ort", Artifact: directory, Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}}}
	scores, err := runtime.Scores(ctx, spec)
	if err != nil {
		t.Fatal(err)
	}
	defer scores.Close()
	input := strings.Repeat("hello ", 32768)
	got, err := scores.Call(ctx, "one", input)
	if err != nil {
		t.Fatal(err)
	}
	if got.Input.ProcessedTokens != 32768 || got.Input.Truncated || got.Scores[1] < .73 || got.Scores[1] > .74 {
		t.Fatalf("incorrect actual sigmoid/context: %+v", got)
	}
	windowSpec := spec
	windowSpec.Deployment.Input.MaxTokens = 65536
	request := tasks.TextWindowsRequest{Text: strings.Repeat("hello ", 42000), Size: 32768, Overlap: 16383}
	windows, err := runtime.ScoreWindows(ctx, windowSpec, request)
	if err != nil {
		t.Fatal(err)
	}
	defer windows.Close()
	out, err := windows.Call(ctx, "one", request)
	if err != nil {
		t.Fatal(err)
	}
	if out.ContentTokens != 42000 || out.Input.ProcessedTokens != 42000 || len(out.Windows) < 2 {
		t.Fatalf("incomplete actual window coverage: %+v", out)
	}
	if _, err = scores.Call(ctx, "one", input+"hello"); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("32769 tokens accepted: %v", err)
	}
	invalid := spec
	invalid.Binding.Contract = config.RemoteClassifierContractLabelDistribution
	if model, loadErr := runtime.Sequence(ctx, invalid); loadErr == nil {
		model.Close()
		t.Fatal("independent ONNX head loaded as categorical")
	}
}
