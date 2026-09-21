package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
)

// The compatibility model has no production registry revision. Freeze only this
// test inventory to the public HF commit verified on 2026-09-16; production
// downloads keep their existing policy. A registered revision takes precedence.
const multimodalCompatibilityRevision = "fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4"

func multimodalAssets(provider, output string) (manifest, []modeldownload.ModelSpec, error) {
	result := manifest{Provider: provider}
	if provider != "candle" {
		return result, nil, fmt.Errorf("multimodal regression requires the candle provider")
	}
	model := config.GetModelByPath("models/mom-embedding-multimodal")
	if model == nil {
		return result, nil, fmt.Errorf("multimodal model is missing from the registry")
	}
	revision := model.Revision
	if revision == "" {
		revision = multimodalCompatibilityRevision
	}
	if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(revision) {
		return result, nil, fmt.Errorf("multimodal model has no immutable release")
	}
	path, err := filepath.Abs(filepath.Join(output, filepath.Base(model.LocalPath), provider, revision))
	if err != nil {
		return result, nil, err
	}
	result.Models = []artifact{{"Multimodal", path, model.RepoID, revision, "MULTIMODAL_MODEL_PATH"}}
	spec := modeldownload.ModelSpec{
		LocalPath: path, RepoID: model.RepoID, Revision: revision, Strict: true,
		// The Candle multimodal loader opens this single root safetensors file,
		// not a sharded checkpoint or the separately exported ONNX encoders.
		RequiredFiles:   []string{"config.json", "tokenizer.json", "model.safetensors"},
		ExcludePatterns: slices.Clone(model.DownloadExcludePatterns),
	}
	spec.ExcludePatterns = append(spec.ExcludePatterns, "*.pt", "*.bin", "onnx/*", "*.onnx", "*.onnx.data", "*.onnx_data")
	return result, []modeldownload.ModelSpec{spec}, nil
}
