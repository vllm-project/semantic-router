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

func TestOwnedTokenWindowsExecuteGloballyDecodedSpan(t *testing.T) {
	for _, provider := range []string{"candle", "ort"} {
		t.Run(provider, func(t *testing.T) {
			var directory string
			if provider == "candle" {
				directory = nativeHeadlessFullFixture(t, 1)
			} else {
				if os.Getenv("ORT_DYLIB_PATH") == "" {
					t.Skip("requires local ONNX Runtime for the synthetic graph")
				}
				directory = t.TempDir()
				fixture := filepath.Join("..", "..", "..", "..", "..", "onnx-binding", "instance", "testdata", "token")
				for _, name := range []string{"config.json", "tokenizer.json", "model.onnx"} {
					data, err := os.ReadFile(filepath.Join(fixture, name))
					if err != nil {
						t.Fatal(err)
					}
					if err = os.WriteFile(filepath.Join(directory, name), data, 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			path := filepath.Join(directory, "config.json")
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var metadata map[string]any
			if err = json.Unmarshal(raw, &metadata); err != nil {
				t.Fatal(err)
			}
			metadata["max_position_embeddings"] = 1024
			metadata["id2label"] = map[string]string{"0": "O", "1": "I-SECRET"}
			metadata["label2id"] = map[string]int{"O": 0, "I-SECRET": 1}
			raw, err = json.Marshal(metadata)
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(path, raw, 0o600); err != nil {
				t.Fatal(err)
			}
			spec := config.ResolvedModelBinding{Recipe: "one", Name: "pii_classifier", Binding: config.ModelBinding{Deployment: "fixture", Adapter: "modernbert", Contract: config.RemoteClassifierContractTokenSpans}, Deployment: config.ModelDeployment{Artifact: directory, Provider: provider, Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 1024, Overflow: "window"}}}
			window := tasks.TextWindowsRequest{Text: strings.Repeat("hello ", 600) + "猫", Size: 128, Overlap: 63}
			task, err := New(nil).TokenWindows(context.Background(), spec, window)
			if err != nil {
				t.Fatal(err)
			}
			defer task.Close()
			out, err := task.Call(context.Background(), "one", window)
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Windows) < 2 || len(out.Result.Entities) != 1 || out.Result.Entities[0].Text != window.Text || out.Result.Entities[0].End != len(window.Text) || out.Result.Input.Truncated {
				t.Fatalf("incomplete global result: windows=%d spans=%d usage=%+v", len(out.Windows), len(out.Result.Entities), out.Result.Input)
			}
			if out.Result.Input.ProcessedTokens != out.Result.Input.OriginalTokens {
				t.Fatal("overlap double counted document usage")
			}
			if _, err = task.Call(context.Background(), "other", window); !errors.Is(err, binding.ErrCapability) {
				t.Fatalf("foreign consumer admitted: %v", err)
			}
			changed := window
			changed.Size = 64
			if _, err = task.Call(context.Background(), "one", changed); !errors.Is(err, binding.ErrInvalidInput) {
				t.Fatalf("changed geometry admitted: %v", err)
			}
			changed = window
			changed.Text = strings.Repeat("hello ", 1025)
			if _, err = task.Call(context.Background(), "one", changed); !errors.Is(err, binding.ErrInputLimit) {
				t.Fatalf("oversized document admitted: %v", err)
			}
			if err = task.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = task.Call(context.Background(), "one", window); !errors.Is(err, binding.ErrClosed) {
				t.Fatalf("closed owner reused: %v", err)
			}
		})
	}
}
