package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
)

func TestAssetSelection(t *testing.T) {
	for suite, count := range map[string]int{"runtime": 10, "perf": 4, "openvino": 2} {
		for _, provider := range []string{"candle", "ort"} {
			m, specs, err := assets(suite, provider, t.TempDir())
			if err != nil || len(m.Models) != count || len(specs) != count {
				t.Fatalf("%s/%s: models=%d specs=%d err=%v", suite, provider, len(m.Models), len(specs), err)
			}
			for i, spec := range specs {
				if filepath.Base(spec.LocalPath) != spec.Revision || filepath.Base(filepath.Dir(spec.LocalPath)) != provider || !strings.Contains(spec.LocalPath, "Vela-") {
					t.Fatalf("artifact is not isolated by runtime and revision: %s", spec.LocalPath)
				}
				if !spec.Strict || spec.Revision != m.Models[i].Revision || spec.RepoID != m.Models[i].RepoID || spec.CheckONNX != (provider == "ort") {
					t.Fatalf("incomplete artifact contract: %+v", spec)
				}
				if m.Models[i].Name == "Hazard" {
					if !slices.Contains(spec.RequiredFiles, "model.safetensors") || !slices.Contains(spec.RequiredFiles, "operating_point.json") || slices.Contains(spec.ExcludePatterns, "*.safetensors") {
						t.Fatalf("operating point lost its source checkpoint identity: %+v", spec)
					}
				}
			}
		}
	}
	for _, args := range [][2]string{{"missing", "candle"}, {"runtime", "cuda"}, {"multimodal", "ort"}} {
		if _, _, err := assets(args[0], args[1], t.TempDir()); err == nil {
			t.Fatalf("accepted unsupported selection: %v", args)
		}
	}
}

func TestMultimodalAssets(t *testing.T) {
	m, specs, err := assets("multimodal", "candle", t.TempDir())
	if err != nil || len(m.Models) != 1 || len(specs) != 1 {
		t.Fatalf("models=%d specs=%d err=%v", len(m.Models), len(specs), err)
	}
	model, spec := m.Models[0], specs[0]
	registered := config.GetModelByPath("models/mom-embedding-multimodal")
	expectedRevision := registered.Revision
	if expectedRevision == "" {
		expectedRevision = multimodalCompatibilityRevision
	}
	if model.Name != "Multimodal" || model.Env != "MULTIMODAL_MODEL_PATH" || model.RepoID != registered.RepoID {
		t.Fatalf("wrong multimodal manifest: %+v", model)
	}
	if model.Revision != expectedRevision || spec.Revision != model.Revision || spec.RepoID != model.RepoID || !spec.Strict || spec.CheckONNX {
		t.Fatalf("incomplete pinned Candle contract: %+v", spec)
	}
	if !filepath.IsAbs(model.Path) || model.Path != spec.LocalPath || filepath.Base(model.Path) != model.Revision || filepath.Base(filepath.Dir(model.Path)) != "candle" {
		t.Fatalf("artifact is not isolated by provider and revision: %s", model.Path)
	}
	for _, unused := range []string{"model.pt", "onnx/audio_encoder.onnx", "onnx/image_encoder.onnx.data"} {
		if !slices.ContainsFunc(spec.ExcludePatterns, func(pattern string) bool {
			matched, matchErr := filepath.Match(pattern, unused)
			return matchErr == nil && matched
		}) {
			t.Errorf("unused multimodal artifact is not excluded: %s", unused)
		}
	}
	if mkdirErr := os.MkdirAll(model.Path, 0o755); mkdirErr != nil {
		t.Fatal(mkdirErr)
	}
	for _, name := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if writeErr := os.WriteFile(filepath.Join(model.Path, name), []byte("fixture"), 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	complete, err := modeldownload.IsModelComplete(model.Path, spec.RequiredFiles)
	if err != nil || !complete {
		t.Fatalf("complete native layout rejected: %v", err)
	}
	for _, name := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		path := filepath.Join(model.Path, name)
		if err := os.Rename(path, path+".absent"); err != nil {
			t.Fatal(err)
		}
		complete, err := modeldownload.IsModelComplete(model.Path, spec.RequiredFiles)
		if err != nil || complete {
			t.Fatalf("missing %s accepted: complete=%v err=%v", name, complete, err)
		}
		if err := os.Rename(path+".absent", path); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMultimodalRegistryRevisionTakesPrecedence(t *testing.T) {
	model := config.GetModelByPath("models/mom-embedding-multimodal")
	original := model.Revision
	t.Cleanup(func() { model.Revision = original })
	model.Revision = strings.Repeat("a", 40)
	m, _, err := assets("multimodal", "candle", t.TempDir())
	if err != nil || len(m.Models) != 1 || m.Models[0].Revision != model.Revision {
		t.Fatalf("registered pin was ignored: %+v err=%v", m, err)
	}
	model.Revision = "main"
	if _, _, err := assets("multimodal", "candle", t.TempDir()); err == nil {
		t.Fatal("mutable registered revision was accepted")
	}
}

func TestRISCVAssetsUseOnlyRegisteredCandleDomain(t *testing.T) {
	m, specs, err := assets("riscv", "candle", t.TempDir())
	if err != nil || len(m.Models) != 1 || len(specs) != 1 {
		t.Fatalf("incomplete RISC-V artifact selection: %+v %v", m, err)
	}
	model := config.GetModelByPath(config.DefaultSystemModels().DomainClassifier)
	if m.Models[0].Name != "Domain" || m.Models[0].RepoID != model.RepoID || m.Models[0].Revision != model.Revision || !specs[0].Strict || specs[0].CheckONNX {
		t.Fatalf("RISC-V did not use the registered Candle checkpoint: %+v", m)
	}
	if !slices.Contains(specs[0].RequiredFiles, "category_mapping.json") {
		t.Fatal("RISC-V router must use the checkpoint's category mapping")
	}
	if _, _, err := assets("riscv", "ort", t.TempDir()); err == nil {
		t.Fatal("RISC-V accepted an unqualified ORT runtime")
	}
}
