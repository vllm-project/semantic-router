//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func layeredEmbeddingFixture(t *testing.T) Options {
	t.Helper()
	options := specialTokenFixture(t, "embedding")
	graph, err := os.ReadFile(filepath.Join(options.ModelPath, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	for _, layer := range []string{"layer-1", "layer-2"} {
		directory := filepath.Join(options.ModelPath, "onnx", layer)
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, "model.onnx"), graph, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(options.ModelPath, "onnx", "model_config.json"), []byte(`{"available_layers":[1,2]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	options.ModelFile = filepath.Join("onnx", "layer-2", "model.onnx")
	return options
}

func TestOwnedEmbeddingLayerInventoryUsesLoadedGraphsWithoutDuplicatePrimary(t *testing.T) {
	options := layeredEmbeddingFixture(t)
	model, loadErr := LoadEmbeddingModel(options)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer model.Close()
	info, infoErr := model.Info()
	if infoErr != nil {
		t.Fatal(infoErr)
	}
	if len(info.Sessions) != 2 || !reflect.DeepEqual(info.AvailableLayers, []int{1, 2}) {
		t.Fatalf("primary graph was loaded twice or layer inventory was guessed: %+v", info)
	}
	for _, layer := range []int{0, 1, 2} {
		result, err := model.Encode("hello", layer, 2)
		assertEmbedding(t, result, err)
	}
	// The primary graph is layer 2, regardless of the architecture metadata.
	// Remove layer 1: it must disappear from capabilities and fail exact lookup.
	if err := os.RemoveAll(filepath.Join(options.ModelPath, "onnx", "layer-1")); err != nil {
		t.Fatal(err)
	}
	isolated, loadErr := LoadEmbeddingModel(options)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer isolated.Close()
	info, infoErr = isolated.Info()
	if infoErr != nil || len(info.Sessions) != 1 || !reflect.DeepEqual(info.AvailableLayers, []int{2}) {
		t.Fatalf("unloaded architectural layer advertised: %+v, %v", info, infoErr)
	}
	if _, err := isolated.Encode("hello", 1, 2); err == nil {
		t.Fatal("missing layer silently used the primary graph")
	}
}

func TestMIGraphXOwnedEmbeddingUsesExplicitMaskedExecutionBudget(t *testing.T) {
	if os.Getenv("ORT_TEST_MIGRAPHX") != "1" {
		t.Skip("requires a real MIGraphX runtime and device")
	}
	options := layeredEmbeddingFixture(t)
	options.Provider, options.MaxInputTokens = "migraphx", 0
	if model, err := LoadEmbeddingModel(options); err == nil {
		_ = model.Close()
		t.Fatal("MIGraphX embedding accepted an undeclared execution budget")
	}
	options.MaxInputTokens = 8
	options.ProfilePrefix = filepath.Join(t.TempDir(), "embedding-fixed-budget")
	path := filepath.Join(options.ModelPath, "config.json")
	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	config["pad_token_id"] = 99
	data, encodeErr := json.Marshal(config)
	if encodeErr != nil {
		t.Fatal(encodeErr)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadEmbeddingModel(options)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer model.Close()
	for _, layer := range []int{1, 2} {
		for _, input := range []struct {
			text   string
			sum    float64
			tokens int
		}{{"hello", 5, 3}, {"hello world", 7, 4}, {"hello", 5, 3}} {
			result, err := model.Encode(input.text, layer, 2)
			assertEmbedding(t, result, err)
			second := input.sum + float64(input.tokens)
			want := input.sum / math.Sqrt(input.sum*input.sum+second*second)
			if math.Abs(float64(result.Values[0])-want) > 1e-6 || result.Input == nil || result.Input.OriginalTokens != input.tokens || result.Input.ProcessedTokens != input.tokens || result.Input.Truncated {
				t.Fatalf("padding changed masked pooling or real-token usage: %+v", result)
			}
		}
	}
	paths, profileErr := model.FinishProfiling()
	if profileErr != nil {
		t.Fatal(profileErr)
	}
	if len(paths) != 2 {
		t.Fatalf("duplicate or missing execution graphs: %v", paths)
	}
	assertStrictGPUProfiles(t, paths)
}
