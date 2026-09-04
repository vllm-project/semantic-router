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

const (
	testQwen3ModelPath      = "models/mom-embedding-pro"
	testGemmaModelPath      = "models/mom-embedding-flash"
	testMultiModalModelPath = "models/mom-embedding-multimodal"
)

func newCandleEmbeddingConfig() *config.RouterConfig {
	return &config.RouterConfig{
		MoMRegistry: map[string]string{
			testEmbeddingModelPath:  testEmbeddingRepoID,
			testQwen3ModelPath:      "llm-semantic-router/mom-embedding-pro",
			testGemmaModelPath:      "llm-semantic-router/mom-embedding-flash",
			testMultiModalModelPath: "llm-semantic-router/mom-embedding-multimodal",
		},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath:     testEmbeddingModelPath,
				Qwen3ModelPath:      testQwen3ModelPath,
				GemmaModelPath:      testGemmaModelPath,
				MultiModalModelPath: testMultiModalModelPath,
			},
		},
	}
}

// TestBuildModelSpecsRequiresCandleRuntimeFilesPerModel guards #2531: every candle
// embedding path (qwen3, gemma, multimodal), not only mmbert (#2195), must require the
// files its runtime hard-loads so partial downloads self-heal. Gemma additionally
// hard-loads the 2_Dense/3_Dense bottleneck weights.
func TestBuildModelSpecsRequiresCandleRuntimeFilesPerModel(t *testing.T) {
	specs, err := BuildModelSpecs(newCandleEmbeddingConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	cases := []struct {
		path string
		want []string
	}{
		{testEmbeddingModelPath, []string{"config.json", "model.safetensors", "tokenizer.json"}},
		{testQwen3ModelPath, []string{"config.json", "model.safetensors", "tokenizer.json"}},
		{testGemmaModelPath, []string{
			"config.json", "model.safetensors", "tokenizer.json",
			"2_Dense/model.safetensors", "3_Dense/model.safetensors",
		}},
		{testMultiModalModelPath, []string{"config.json", "model.safetensors", "tokenizer.json"}},
	}

	for _, tc := range cases {
		spec, ok := findSpecByPath(specs, tc.path)
		if !ok {
			t.Fatalf("BuildModelSpecs() did not produce a spec for %q; got %#v", tc.path, specs)
		}
		for _, want := range tc.want {
			if !slices.Contains(spec.RequiredFiles, want) {
				t.Errorf("spec %q RequiredFiles = %#v, missing %q", tc.path, spec.RequiredFiles, want)
			}
		}
	}
}

// TestOnnxOnlyCandleEmbeddingDirsReportedIncomplete extends the #2172 reproduction to the
// qwen3 and multimodal paths: an ONNX-only directory must read as incomplete so the full
// snapshot is re-downloaded instead of crash-looping at classifier init (#2531).
func TestOnnxOnlyCandleEmbeddingDirsReportedIncomplete(t *testing.T) {
	specs, err := BuildModelSpecs(newCandleEmbeddingConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, path := range []string{testQwen3ModelPath, testMultiModalModelPath} {
		spec, ok := findSpecByPath(specs, path)
		if !ok {
			t.Fatalf("BuildModelSpecs() did not produce a spec for %q", path)
		}

		dir := t.TempDir()
		writeModelFile(t, dir, "config.json", "{}")
		writeModelFile(t, filepath.Join(dir, "onnx", "layer-6"), "model.onnx", "onnx-bytes")

		complete, err := IsModelComplete(dir, spec.RequiredFiles)
		if err != nil {
			t.Fatalf("IsModelComplete(%q) error = %v", path, err)
		}
		if complete {
			t.Errorf("ONNX-only dir for %q reported complete; expected incomplete", path)
		}
	}
}

// TestGemmaDirWithoutDenseWeightsReportedIncomplete guards the gemma-specific slice of
// #2531: root weights and tokenizer alone are not enough, the dense-bottleneck weights the
// runtime hard-loads must be present before the model reads as complete.
func TestGemmaDirWithoutDenseWeightsReportedIncomplete(t *testing.T) {
	specs, err := BuildModelSpecs(newCandleEmbeddingConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, ok := findSpecByPath(specs, testGemmaModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce a spec for %q", testGemmaModelPath)
	}

	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", "{}")
	writeModelFile(t, dir, "model.safetensors", "weights")
	writeModelFile(t, dir, "tokenizer.json", "{}")

	complete, err := IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if complete {
		t.Fatalf("gemma dir without dense weights reported complete; RequiredFiles = %#v", spec.RequiredFiles)
	}

	writeModelFile(t, filepath.Join(dir, "2_Dense"), "model.safetensors", "dense-2")
	writeModelFile(t, filepath.Join(dir, "3_Dense"), "model.safetensors", "dense-3")

	complete, err = IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if !complete {
		t.Fatalf("full gemma dir reported incomplete; RequiredFiles = %#v", spec.RequiredFiles)
	}
}

// TestBuildModelSpecsSkipsCandleEmbeddingModelsForRemoteBackend keeps the remote-backend
// exemption intact for the newly covered paths.
func TestBuildModelSpecsSkipsCandleEmbeddingModelsForRemoteBackend(t *testing.T) {
	cfg := newCandleEmbeddingConfig()
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
