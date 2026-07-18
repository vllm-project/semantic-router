package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	testEmbeddingModelPath = "models/mmbert-embed-32k-2d-matryoshka"
	testEmbeddingRepoID    = "llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
)

func newEmbeddingOnlyConfig() *config.RouterConfig {
	return &config.RouterConfig{
		MoMRegistry: map[string]string{
			testEmbeddingModelPath: testEmbeddingRepoID,
		},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: testEmbeddingModelPath,
			},
		},
	}
}

func findSpecByPath(specs []ModelSpec, localPath string) (ModelSpec, bool) {
	for _, spec := range specs {
		if spec.LocalPath == localPath {
			return spec, true
		}
	}
	return ModelSpec{}, false
}

// TestBuildModelSpecsRequiresEmbeddingModelWeightsAndTokenizer guards #2172:
// the candle embedding runtime loads the model from model.safetensors + tokenizer.json,
// so those must be part of the embedding model's completeness contract. Otherwise a dir
// holding only config.json + onnx/ (the state shipped in the image) passes the
// nested-onnx weight heuristic and the safetensors/tokenizer download is never triggered.
func TestBuildModelSpecsRequiresEmbeddingModelWeightsAndTokenizer(t *testing.T) {
	specs, err := BuildModelSpecs(newEmbeddingOnlyConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce a spec for %q; got %#v", testEmbeddingModelPath, specs)
	}

	for _, want := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
		if !slices.Contains(spec.RequiredFiles, want) {
			t.Fatalf("embedding spec RequiredFiles = %#v, missing %q", spec.RequiredFiles, want)
		}
	}
}

func TestBuildModelSpecsSkipsEmbeddingModelsForRemoteBackend(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.EmbeddingModels.EmbeddingConfig = config.HNSWConfig{
		Backend:   config.EmbeddingBackendOpenAICompatible,
		ModelType: config.EmbeddingModelTypeRemote,
	}
	cfg.EmbeddingModels.Endpoint = config.EmbeddingEndpointConfig{
		BaseURL: "http://embedding-service:8000/v1",
		Model:   "BAAI/bge-m3",
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs for remote embedding backend, want 0: %#v", len(specs), specs)
	}
}

// TestOnnxOnlyEmbeddingDirReportedIncomplete reproduces the #2172 symptom end-to-end:
// an ONNX-only embedding directory (config.json + nested onnx weights, no safetensors /
// tokenizer.json) must be reported as missing so the full snapshot is re-downloaded.
func TestOnnxOnlyEmbeddingDirReportedIncomplete(t *testing.T) {
	specs, err := BuildModelSpecs(newEmbeddingOnlyConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce a spec for %q", testEmbeddingModelPath)
	}

	// Mirror the directory the image actually ships: config.json + onnx/layer-*/model.onnx,
	// but no model.safetensors / tokenizer.json.
	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", "{}")
	writeModelFile(t, dir, "README.md", "# model")
	writeModelFile(t, filepath.Join(dir, "onnx", "layer-6"), "model.onnx", "onnx-bytes")

	complete, err := IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if complete {
		t.Fatalf("ONNX-only embedding dir reported complete; expected incomplete so the full snapshot is re-downloaded")
	}
}

// TestFullEmbeddingDirReportedComplete is the control: once the safetensors weights and
// tokenizer are present, the embedding model is complete and is not re-downloaded.
func TestFullEmbeddingDirReportedComplete(t *testing.T) {
	specs, err := BuildModelSpecs(newEmbeddingOnlyConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, _ := findSpecByPath(specs, testEmbeddingModelPath)

	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", "{}")
	writeModelFile(t, dir, "model.safetensors", "weights")
	writeModelFile(t, dir, "tokenizer.json", "{}")

	complete, err := IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if !complete {
		t.Fatalf("complete embedding dir reported incomplete; RequiredFiles = %#v", spec.RequiredFiles)
	}
}

func writeModelFile(t *testing.T, dir, name, contents string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) error = %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", filepath.Join(dir, name), err)
	}
}
