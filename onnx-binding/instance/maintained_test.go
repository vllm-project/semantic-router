//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Opt-in maintained-checkpoint smoke; this is correctness/provider evidence,
// never a latency benchmark. The caller supplies a revision-locked cache.
func TestMaintainedMIGraphXModels(t *testing.T) {
	root := os.Getenv("ORT_MAINTAINED_MODELS_ROOT")
	if root == "" {
		t.Skip("requires revision-locked maintained model cache")
	}
	opts := Options{Provider: "migraphx", Precision: "native", MaxInputTokens: 128, Overflow: "reject"}
	if os.Getenv("ORT_MAINTAINED_TASK") != "embedding" {
		opts.ModelPath = filepath.Join(root, "mmbert32k-intent-classifier-merged")
		opts.ModelFile = "onnx/model.onnx"
		opts.ProfilePrefix = filepath.Join(t.TempDir(), "sequence")
		model, err := LoadSequenceClassifier(opts)
		if err != nil {
			t.Fatal(err)
		}
		defer model.Close()
		clone, err := model.Clone()
		if err != nil {
			t.Fatal(err)
		}
		defer clone.Close()
		if closeErr := model.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		result, err := clone.Classify("Explain how photosynthesis works in plants.")
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Probabilities) < 2 || len(result.Labels) != len(result.Probabilities) {
			t.Fatalf("invalid maintained distribution: %+v", result)
		}
		paths, err := clone.FinishProfiling()
		if err != nil {
			t.Fatal(err)
		}
		assertStrictGPUProfiles(t, paths)
		t.Logf("maintained sequence: %d labels, processed tokens %d", len(result.Labels), result.Input.ProcessedTokens)
	}
	opts.ModelPath = filepath.Join(root, "mmbert-embed-32k-2d-matryoshka")
	opts.ModelFile = ""
	opts.ProfilePrefix = filepath.Join(t.TempDir(), "embedding")
	embedding, err := LoadEmbeddingModel(opts)
	if err != nil {
		t.Fatal(err)
	}
	defer embedding.Close()
	vector, err := embedding.Encode("Explain how photosynthesis works in plants.", 0, 256)
	if err != nil {
		t.Fatal(err)
	}
	if len(vector.Values) != 256 {
		t.Fatalf("dimension=%d", len(vector.Values))
	}
	paths, err := embedding.FinishProfiling()
	if err != nil {
		t.Fatal(err)
	}
	assertStrictGPUProfiles(t, paths)
	t.Logf("maintained embedding: %d dimensions, processed tokens %d", len(vector.Values), vector.Input.ProcessedTokens)
}

func assertStrictGPUProfiles(t *testing.T, paths []string) {
	t.Helper()
	gpu := 0
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if directory := os.Getenv("ORT_PROFILE_EVIDENCE_DIR"); directory != "" {
			if err := os.MkdirAll(directory, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(directory, filepath.Base(path)), data, 0o600); err != nil {
				t.Fatal(err)
			}
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
				t.Fatal("strict model executed CPU nodes")
			}
			if event.Args.Provider == "MIGraphXExecutionProvider" {
				gpu++
			}
		}

	}
	if gpu == 0 {
		t.Fatal("no profiled MIGraphX execution")
	}
	t.Logf("strict maintained profile: MIGraphX kernel execution records=%d, CPU records=0", gpu)
}
