package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuiltInReleaseSkipsTrainingArtifactsWithoutExcludingRuntimeWeights(t *testing.T) {
	const path = "models/Vela-1.0-Encoder-307M-Feedback"
	const repo = "llm-semantic-router/Vela-1.0-Encoder-307M-Feedback"
	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name: "feedback", Type: "local", ModelPath: path, Labels: []string{"SAT", "NO_FEEDBACK"},
			}}},
		},
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := findSpecByPath(specs, path)
	if !ok {
		t.Fatal("missing release download")
	}
	for _, name := range []string{
		"reproduction/sources/checkpoint/model.safetensors", "reproduction/portable/model.onnx",
		"reproducibility/training/data.jsonl", "lora/adapter_model.safetensors",
	} {
		if !revisionArtifactExcluded(name, spec.ExcludePatterns) {
			t.Fatalf("optional training artifact would be downloaded: %s", name)
		}
	}
	for _, name := range []string{
		"config.json", "model.safetensors", "tokenizer.json", "classification_heads.pt",
		"onnx/layer-22/model.onnx", "onnx/layer-22/model.onnx.data", "onnx/layer-3/model_fa_fp16.onnx",
	} {
		if revisionArtifactExcluded(name, spec.ExcludePatterns) {
			t.Fatalf("runtime artifact was excluded: %s", name)
		}
	}
	args := buildDownloadArgs(spec)
	for _, pattern := range spec.ExcludePatterns {
		if !slices.Contains(args, pattern) {
			t.Fatalf("HF download did not receive exclusion: %s", pattern)
		}
	}
	// Changing repositories must not inherit another publisher's artifact layout.
	cfg.MoMRegistry[path] = "example/custom-feedback"
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	custom, ok := findSpecByPath(specs, path)
	if !ok || custom.RepoID == repo || len(custom.ExcludePatterns) != 0 || custom.Revision != "main" {
		t.Fatalf("custom repository inherited built-in policy: %+v", custom)
	}
}

func TestReleaseExclusionsComposeWithRuntimeAndDoNotLaunderMissingWeights(t *testing.T) {
	const path = "models/Vela-1.0-Encoder-307M-Feedback"
	model := config.GetModelByPath(path)
	runtime := []string{"*.onnx"}
	patterns := modelDownloadExcludePatterns(model.Aliases[0], model.RepoID, runtime)
	if len(runtime) != 1 || !slices.Contains(patterns, "*.onnx") || !slices.Contains(patterns, "reproduction/*") {
		t.Fatalf("runtime/release composition changed its inputs: %v %v", runtime, patterns)
	}
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: model.RepoID, Revision: model.Revision, ExcludePatterns: patterns}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	for _, name := range []string{"reproduction/source/model.safetensors", "lora/adapter_model.safetensors"} {
		writeHFRevisionArtifact(t, spec, name, "optional weights", true)
	}
	if _, err := verifyHFModelRevision(spec); err == nil {
		t.Fatal("training checkpoint established runtime completeness")
	}
	writeHFRevisionArtifact(t, spec, "model.safetensors", "runtime weights", true)
	if _, err := verifyHFModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	// Existing local research files do not require provenance or a download refresh.
	if err := os.WriteFile(filepath.Join(spec.LocalPath, "reproduction/source/model.safetensors"), []byte("local experiment"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyHFModelRevision(spec); err != nil {
		t.Fatal(err)
	}
}
