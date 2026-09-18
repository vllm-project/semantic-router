//go:build !windows && cgo && published_model_tests

package openvino_binding

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
)

type publishedModel struct {
	Name        string `json:"name"`
	RepoID      string `json:"repo_id"`
	Revision    string `json:"revision"`
	IRPath      string `json:"ir_path"`
	ConfigPath  string `json:"config_path"`
	Dimension   int    `json:"dimension"`
	PadTokenID  *int   `json:"pad_token_id"`
	EndTokenIDs []int  `json:"end_token_ids"`
}

func (model publishedModel) ownedOptions(device string) ModelOptions {
	return ModelOptions{
		ModelPath: model.IRPath, Device: device, MaxTokens: 512,
		Overflow: "truncate", PadTokenID: *model.PadTokenID, EndTokenIDs: model.EndTokenIDs,
	}
}

// TestPublishedOpenVINO is an opt-in integration target whose prerequisites are
// required. Missing artifacts, devices or inference results are failures.
func TestPublishedOpenVINO(t *testing.T) {
	device := strings.ToUpper(os.Getenv("OPENVINO_TEST_DEVICE"))
	if device == "" {
		device = "CPU"
	}
	if !slices.Contains(GetAvailableDevices(), device) {
		t.Fatalf("requested OpenVINO device %q is unavailable; devices=%v", device, GetAvailableDevices())
	}
	manifestPath := os.Getenv("OPENVINO_TEST_MANIFEST")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read converted artifact manifest: %v", err)
	}
	var manifest struct {
		Provider string           `json:"provider"`
		Models   []publishedModel `json:"models"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Provider != "openvino" || len(manifest.Models) != 2 {
		t.Fatalf("expected exactly two pinned OpenVINO artifacts: %+v", manifest)
	}
	byName := map[string]publishedModel{}
	for _, model := range manifest.Models {
		if model.Name != "Domain" && model.Name != "Embedding" || len(model.Revision) != 40 || model.Dimension < 1 || model.PadTokenID == nil || len(model.EndTokenIDs) == 0 {
			t.Fatalf("invalid artifact identity: %+v", model)
		}
		byName[model.Name] = model
	}
	if len(byName) != 2 {
		t.Fatal("Domain and Embedding must both be present")
	}
	t.Run("embedding", func(t *testing.T) {
		model := byName["Embedding"]
		handle, err := LoadEmbeddingModel(model.ownedOptions(device))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handle.Close() })
		texts := []string{
			"How do I reset the password for my account?",
			"I forgot my account password and need to reset it.",
			"Mitochondria generate energy in biological cells.",
		}
		vectors := make([][]float32, len(texts))
		for i, text := range texts {
			result, err := handle.Embed(text)
			if err != nil {
				t.Fatal(err)
			}
			vectors[i] = result.Values
			if len(vectors[i]) != model.Dimension {
				t.Fatalf("embedding dimensions=%d, expected=%d", len(vectors[i]), model.Dimension)
			}
			var norm float64
			for _, value := range vectors[i] {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("nonfinite embedding")
				}
				norm += float64(value) * float64(value)
			}
			if norm <= 0 {
				t.Fatal("zero embedding")
			}
		}
		related, unrelated := embeddingCosine(vectors[0], vectors[1]), embeddingCosine(vectors[0], vectors[2])
		if related <= unrelated {
			t.Fatalf("semantic order reversed: related=%f unrelated=%f", related, unrelated)
		}
		t.Logf("%s@%s: dimension=%d, related=%f, unrelated=%f", model.RepoID, model.Revision, model.Dimension, related, unrelated)
	})
	t.Run("domain", func(t *testing.T) {
		model := byName["Domain"]
		handle, err := LoadClassifierModel(model.ownedOptions(device), model.Dimension)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handle.Close() })
		data, err := os.ReadFile(model.ConfigPath)
		if err != nil {
			t.Fatal(err)
		}
		var config struct {
			Labels map[string]string `json:"id2label"`
		}
		if err := json.Unmarshal(data, &config); err != nil {
			t.Fatal(err)
		}
		for _, probe := range []struct{ text, label string }{
			{"Solve the quadratic equation x squared minus five x plus six equals zero using the quadratic formula.", "math"},
			{"Explain how a binary search tree works and analyze its search algorithm time complexity in Python.", "computer science"},
		} {
			result, err := handle.Classify(probe.text)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Probabilities) != model.Dimension || result.Class < 0 || result.Class >= model.Dimension {
				t.Fatalf("invalid classifier result: %+v", result)
			}
			var sum float64
			for _, probability := range result.Probabilities {
				if math.IsNaN(float64(probability)) || math.IsInf(float64(probability), 0) || probability < 0 || probability > 1 {
					t.Fatalf("invalid probability: %f", probability)
				}
				sum += float64(probability)
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Fatalf("probabilities do not sum to one: %f", sum)
			}
			if actual := config.Labels[strconv.Itoa(result.Class)]; actual != probe.label {
				t.Fatalf("domain=%q, expected=%q, confidence=%f", actual, probe.label, result.Confidence)
			}
			t.Logf("%s@%s: %s confidence=%f", model.RepoID, model.Revision, probe.label, result.Confidence)
		}
	})
	if t.Failed() {
		return
	}
	reportDir := os.Getenv("OPENVINO_TEST_REPORT_DIR")
	if reportDir == "" {
		t.Fatal("OPENVINO_TEST_REPORT_DIR is required for inference evidence")
	}
	receipt, err := json.MarshalIndent(map[string]any{
		"provider": "openvino", "device": device, "passed": true, "models": manifest.Models,
		"platform":        runtime.GOOS + "/" + runtime.GOARCH,
		"runtime_version": GetVersion(),
		"contracts":       []string{"embedding-semantic-order", "domain-labels-and-probabilities"},
	}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(reportDir, "inference.json"), append(receipt, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func embeddingCosine(left, right []float32) float64 {
	var dot, normLeft, normRight float64
	for i := range left {
		x, y := float64(left[i]), float64(right[i])
		dot += x * y
		normLeft += x * x
		normRight += y * y
	}
	return dot / math.Sqrt(normLeft*normRight)
}
