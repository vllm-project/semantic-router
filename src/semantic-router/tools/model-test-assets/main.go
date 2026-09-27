// Command model-test-assets provisions the registered artifacts exercised by
// model and performance tests. It does not change the router's on-demand downloads.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
)

type artifact struct {
	Name     string `json:"name"`
	Path     string `json:"path"`
	RepoID   string `json:"repo_id"`
	Revision string `json:"revision"`
	Env      string `json:"env"`
}

type manifest struct {
	Provider string     `json:"provider"`
	Models   []artifact `json:"models"`
}

func main() {
	provider := flag.String("provider", "candle", "artifact format: candle or ort")
	suite := flag.String("suite", "runtime", "runtime, perf, openvino, riscv, or multimodal")
	output := flag.String("output", "models", "model directory")
	manifestPath := flag.String("manifest", "", "write artifact identities as JSON")
	download := flag.Bool("download", false, "download and verify required artifacts")
	flag.Parse()
	result, specs, err := assets(*suite, *provider, *output)
	if err == nil && *download {
		err = provision(specs)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err == nil && *manifestPath != "" {
		err = os.MkdirAll(filepath.Dir(*manifestPath), 0o755)
		if err == nil {
			err = os.WriteFile(*manifestPath, append(data, '\n'), 0o600)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *manifestPath == "" {
		fmt.Println(string(data))
	}
}

func assets(suite, provider, output string) (manifest, []modeldownload.ModelSpec, error) {
	result := manifest{Provider: provider}
	if provider != "candle" && provider != "ort" {
		return result, nil, fmt.Errorf("unsupported model provider %q", provider)
	}
	if suite == "multimodal" {
		return multimodalAssets(provider, output)
	}
	if !slices.Contains([]string{"runtime", "perf", "openvino", "riscv"}, suite) {
		return result, nil, fmt.Errorf("unknown model suite %q", suite)
	}
	if suite == "riscv" && provider != "candle" {
		return result, nil, fmt.Errorf("RISC-V emulation only qualifies Candle")
	}
	defaults := config.DefaultGlobalConfig()
	system := config.DefaultSystemModels()
	paths := map[string]string{
		"Domain": system.DomainClassifier, "Guard": system.PromptGuard,
		"PII": system.PIIClassifier, "FactCheck": system.FactCheckClassifier,
		"Feedback": system.FeedbackDetector, "Safety": system.Safety,
		"Hazard": system.Hazard, "Embedding": defaults.MmBertModelPath,
	}
	// These optional task heads have no global default module. Resolve their
	// release from the same registry that provisions explicit deployments.
	for name, purpose := range map[string]config.ModelPurpose{"Modality": config.PurposeModalityDetection, "Reranker": config.PurposeReranking} {
		for _, model := range config.GetModelsByPurpose(purpose) {
			if slices.Contains(model.Tags, "vela") {
				if paths[name] != "" {
					return result, nil, fmt.Errorf("ambiguous maintained %s model", name)
				}
				paths[name] = model.LocalPath
			}
		}
	}
	var specs []modeldownload.ModelSpec
	for _, name := range []string{"Domain", "Guard", "PII", "FactCheck", "Feedback", "Modality", "Safety", "Hazard", "Embedding", "Reranker"} {
		if suite == "perf" && !slices.Contains([]string{"Domain", "Guard", "PII", "Embedding"}, name) ||
			suite == "openvino" && name != "Domain" && name != "Embedding" ||
			suite == "riscv" && name != "Domain" {
			continue
		}
		model := config.GetModelByPath(paths[name])
		if model == nil || !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(model.Revision) {
			return result, nil, fmt.Errorf("%s has no immutable registered release", name)
		}
		path, err := filepath.Abs(filepath.Join(output, filepath.Base(model.LocalPath), provider, model.Revision))
		if err != nil {
			return result, nil, err
		}
		envName := "VLLM_SR_" + strings.ToUpper(name) + "_MODEL"
		if name == "Guard" {
			envName = "VLLM_SR_JAILBREAK_MODEL"
		}
		result.Models = append(result.Models, artifact{name, path, model.RepoID, model.Revision, envName})
		spec := modeldownload.ModelSpec{
			LocalPath: path, RepoID: model.RepoID, Revision: model.Revision,
			RequiredFiles: []string{"config.json", "tokenizer.json"}, Strict: true,
			ExcludePatterns: slices.Clone(model.DownloadExcludePatterns),
		}
		if provider == "candle" {
			spec.RequiredFileGroups = [][]string{{"model.safetensors", "model.safetensors.index.json"}}
			spec.ExcludePatterns = append(spec.ExcludePatterns, "*.onnx", "*.onnx.data", "*.onnx_data", "onnx/weights.data")
		} else {
			spec.CheckONNX = true
			spec.RequiredFileGroups = [][]string{{"onnx/model.onnx", "model.onnx", "onnx/encoder.onnx"}}
			if name != "Hazard" {
				spec.ExcludePatterns = append(spec.ExcludePatterns, "*.safetensors", "*.bin")
			}
		}
		if name == "Hazard" {
			// The frozen operating point verifies its native source checkpoint
			// as well as the selected graph, including for ONNX execution.
			spec.RequiredFiles = append(spec.RequiredFiles, "model.safetensors", "operating_point.json")
		}
		for _, mapping := range []string{defaults.CategoryMappingPath, defaults.PromptGuard.JailbreakMappingPath, defaults.PIIMappingPath} {
			if filepath.Dir(mapping) == model.LocalPath {
				spec.RequiredFiles = append(spec.RequiredFiles, filepath.Base(mapping))
			}
		}
		specs = append(specs, spec)
	}
	if suite == "runtime" {
		return runtimeExtensions(result, specs, output)
	}
	return result, specs, nil
}

func provision(specs []modeldownload.ModelSpec) error {
	missing, err := modeldownload.GetMissingModels(specs)
	if err != nil {
		return err
	}
	if len(missing) == 0 {
		return nil
	}
	if err := modeldownload.CheckHuggingFaceCLI(); err != nil {
		return err
	}
	for _, spec := range missing {
		if err := modeldownload.DownloadModel(spec, modeldownload.GetDownloadConfig()); err != nil {
			return fmt.Errorf("%s: %w", spec.RepoID, err)
		}
	}
	return nil
}
