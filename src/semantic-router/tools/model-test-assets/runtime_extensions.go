package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
)

// Runtime qualification is an explicit supported inventory, not a Cartesian
// product of every artifact and provider. Halu's optional ORT reference export
// and maximum-context Omni qualification retain their separate explicit inputs.
func runtimeExtensions(result manifest, specs []modeldownload.ModelSpec, output string) (manifest, []modeldownload.ModelSpec, error) {
	names := []struct{ name, path, env string }{
		{"Halu", "models/Vela-1.0-Encoder-307M-Halu", "VLLM_SR_HALU_MODEL"},
	}
	if result.Provider == "ort" {
		names = []struct{ name, path, env string }{
			{"OmniNano", "models/vela-1.0-omni-nano", "VLLM_SR_OMNI_NANO_MODEL"},
			{"OmniMini", "models/vela-1.0-omni-mini", "VLLM_SR_OMNI_MINI_MODEL"},
		}
	}
	for _, item := range names {
		model := config.GetModelByPath(item.path)
		if model == nil || !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(model.Revision) {
			return result, nil, fmt.Errorf("%s has no immutable registered release", item.name)
		}
		path := filepath.Join(output, filepath.Base(model.LocalPath), result.Provider, model.Revision)
		if model.PreparedArtifact != "" {
			path = filepath.Join(output, "vela-omni-artifacts", model.ArtifactBundle)
		}
		path, err := filepath.Abs(path)
		if err != nil {
			return result, nil, err
		}
		result.Models = append(result.Models, artifact{item.name, path, model.RepoID, model.Revision, item.env})
		spec := modeldownload.ModelSpec{
			LocalPath: path, RepoID: model.RepoID, Revision: model.Revision, Strict: true,
			PreparedArtifact: model.PreparedArtifact, ArtifactBundle: model.ArtifactBundle,
		}
		if model.PreparedArtifact == "" {
			spec.RequiredFiles = []string{"config.json", "tokenizer.json", "operating_point.json"}
			spec.RequiredFileGroups = [][]string{{"model.safetensors", "model.safetensors.index.json"}}
			spec.ExcludePatterns = append(slices.Clone(model.DownloadExcludePatterns), "*.onnx", "*.onnx.data", "*.onnx_data")
		}
		specs = append(specs, spec)
	}
	return result, specs, nil
}
