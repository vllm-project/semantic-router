package trainingartifacts

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type ModelEvalScript string

const (
	ModelEvalSignalEvalScript     ModelEvalScript = "signal_eval"
	ModelEvalMMLUProVLLMEval      ModelEvalScript = "mmlu_pro_vllm_eval"
	ModelEvalResultToConfigScript ModelEvalScript = "result_to_config"
)

type RuntimeSelectionService string

const (
	RuntimeSelectionAutoMixVerifier RuntimeSelectionService = "automix_verifier"
	RuntimeSelectionRouterR1Server  RuntimeSelectionService = "router_r1_server"
)

type Contract struct {
	Version          string                   `json:"version"`
	ModelEval        ModelEvalContract        `json:"model_eval"`
	MLPipeline       MLPipelineContract       `json:"ml_pipeline"`
	RuntimeSelection RuntimeSelectionContract `json:"runtime_selection"`
	RuntimeDefaults  RuntimeDefaultsContract  `json:"runtime_defaults"`
}

type ModelEvalContract struct {
	RepoRelativeDir string           `json:"repo_relative_dir"`
	Scripts         ModelEvalScripts `json:"scripts"`
	Outputs         ModelEvalOutputs `json:"outputs"`
}

type ModelEvalScripts struct {
	SignalEval      string `json:"signal_eval"`
	MMLUProVLLMEval string `json:"mmlu_pro_vllm_eval"`
	ResultToConfig  string `json:"result_to_config"`
}

type ModelEvalOutputs struct {
	DefaultConfigOutputFile string `json:"default_config_output_file"`
	SignalEvalPrefix        string `json:"signal_eval_prefix"`
	SignalEvalExtension     string `json:"signal_eval_extension"`
	SystemAccuracyDirName   string `json:"system_accuracy_dir_name"`
}

type MLPipelineContract struct {
	RepoRelativeDir string            `json:"repo_relative_dir"`
	Scripts         MLPipelineScripts `json:"scripts"`
	Outputs         MLPipelineOutputs `json:"outputs"`
}

type MLPipelineScripts struct {
	Benchmark string `json:"benchmark"`
	Server    string `json:"server"`
}

type MLPipelineOutputs struct {
	BenchmarkOutputFile string               `json:"benchmark_output_file"`
	TrainOutputDirName  string               `json:"train_output_dir_name"`
	GeneratedValuesFile string               `json:"generated_values_file"`
	CacheDirName        string               `json:"cache_dir_name"`
	ModelFiles          MLPipelineModelFiles `json:"model_files"`
}

type MLPipelineModelFiles struct {
	KNN    string `json:"knn"`
	KMeans string `json:"kmeans"`
	SVM    string `json:"svm"`
	MLP    string `json:"mlp"`
}

type RuntimeSelectionContract struct {
	Services RuntimeSelectionServices `json:"services"`
}

type RuntimeSelectionServices struct {
	AutoMixVerifier TrainingServiceContract `json:"automix_verifier"`
	RouterR1Server  TrainingServiceContract `json:"router_r1_server"`
}

type TrainingServiceContract struct {
	DisplayName        string `json:"display_name"`
	RepoRelativeScript string `json:"repo_relative_script"`
	ConfigKey          string `json:"config_key"`
}

type RuntimeDefaultsContract struct {
	Providers         map[string]any  `json:"providers"`
	Embeddings        map[string]any  `json:"embeddings"`
	SemanticCache     map[string]any  `json:"semantic_cache"`
	Tools             map[string]any  `json:"tools"`
	PromptGuard       map[string]any  `json:"prompt_guard"`
	DomainClassifier  map[string]any  `json:"domain_classifier"`
	PIIClassifier     map[string]any  `json:"pii_classifier"`
	CategoryReasoning map[string]bool `json:"category_reasoning"`
}

//go:embed contract.json
var rawContract []byte

var currentContract = mustLoadContract()

func mustLoadContract() Contract {
	var contract Contract
	if err := json.Unmarshal(rawContract, &contract); err != nil {
		panic(fmt.Sprintf("trainingartifacts: failed to parse contract: %v", err))
	}
	if err := contract.Validate(); err != nil {
		panic(fmt.Sprintf("trainingartifacts: invalid contract: %v", err))
	}
	return contract
}

func CurrentContract() Contract {
	return currentContract
}

func (c Contract) Validate() error {
	requiredStrings := map[string]string{
		"version":                                c.Version,
		"model_eval.repo_relative_dir":           c.ModelEval.RepoRelativeDir,
		"model_eval.scripts.signal_eval":         c.ModelEval.Scripts.SignalEval,
		"model_eval.scripts.mmlu_pro_vllm_eval":  c.ModelEval.Scripts.MMLUProVLLMEval,
		"model_eval.scripts.result_to_config":    c.ModelEval.Scripts.ResultToConfig,
		"model_eval.outputs.default_config_file": c.ModelEval.Outputs.DefaultConfigOutputFile,
		"ml_pipeline.repo_relative_dir":          c.MLPipeline.RepoRelativeDir,
		"ml_pipeline.scripts.benchmark":          c.MLPipeline.Scripts.Benchmark,
		"ml_pipeline.scripts.server":             c.MLPipeline.Scripts.Server,
		"ml_pipeline.outputs.benchmark_output":   c.MLPipeline.Outputs.BenchmarkOutputFile,
		"ml_pipeline.outputs.train_output_dir":   c.MLPipeline.Outputs.TrainOutputDirName,
		"ml_pipeline.outputs.generated_values":   c.MLPipeline.Outputs.GeneratedValuesFile,
		"ml_pipeline.outputs.cache_dir":          c.MLPipeline.Outputs.CacheDirName,
	}
	for key, value := range requiredStrings {
		if value == "" {
			return fmt.Errorf("%s must not be empty", key)
		}
	}

	if c.RuntimeSelection.Services.AutoMixVerifier.DisplayName == "" ||
		c.RuntimeSelection.Services.AutoMixVerifier.RepoRelativeScript == "" ||
		c.RuntimeSelection.Services.AutoMixVerifier.ConfigKey == "" {
		return fmt.Errorf("runtime_selection.services.automix_verifier must be fully defined")
	}
	if c.RuntimeSelection.Services.RouterR1Server.DisplayName == "" ||
		c.RuntimeSelection.Services.RouterR1Server.RepoRelativeScript == "" ||
		c.RuntimeSelection.Services.RouterR1Server.ConfigKey == "" {
		return fmt.Errorf("runtime_selection.services.router_r1_server must be fully defined")
	}
	if len(c.RuntimeDefaults.CategoryReasoning) == 0 {
		return fmt.Errorf("runtime_defaults.category_reasoning must not be empty")
	}
	return nil
}

func contractJoin(projectRoot, repoRelativePath string) string {
	return filepath.Join(projectRoot, filepath.FromSlash(repoRelativePath))
}

func (c Contract) ModelEvalDir(projectRoot string) string {
	return contractJoin(projectRoot, c.ModelEval.RepoRelativeDir)
}

func (c Contract) MLPipelineDir(projectRoot string) string {
	return contractJoin(projectRoot, c.MLPipeline.RepoRelativeDir)
}

func (c Contract) ModelEvalScriptPath(projectRoot string, script ModelEvalScript) (string, error) {
	scriptName, err := c.ModelEval.Scripts.ScriptName(script)
	if err != nil {
		return "", err
	}
	return filepath.Join(c.ModelEvalDir(projectRoot), scriptName), nil
}

func (c Contract) SignalEvalOutputPath(outputDir, datasetID string) string {
	return filepath.Join(
		outputDir,
		c.ModelEval.Outputs.SignalEvalPrefix+datasetID+c.ModelEval.Outputs.SignalEvalExtension,
	)
}

func (c Contract) SystemAccuracyOutputDir(outputDir string) string {
	return filepath.Join(outputDir, c.ModelEval.Outputs.SystemAccuracyDirName)
}

func (c Contract) BenchmarkScriptPath(trainingDir string) string {
	return filepath.Join(trainingDir, c.MLPipeline.Scripts.Benchmark)
}

func (c Contract) BenchmarkOutputPath(jobDir string) string {
	return filepath.Join(jobDir, c.MLPipeline.Outputs.BenchmarkOutputFile)
}

func (c Contract) TrainOutputDir(dataDir string) string {
	return filepath.Join(dataDir, c.MLPipeline.Outputs.TrainOutputDirName)
}

func (c Contract) CacheDir(jobDir string) string {
	return filepath.Join(jobDir, c.MLPipeline.Outputs.CacheDirName)
}

func (c Contract) GeneratedValuesPath(jobDir string) string {
	return filepath.Join(jobDir, c.MLPipeline.Outputs.GeneratedValuesFile)
}

func (c Contract) ModelArtifactPath(modelsPath, algorithm string) (string, error) {
	fileName, err := c.MLPipeline.Outputs.ModelFiles.FileName(algorithm)
	if err != nil {
		return "", err
	}
	return filepath.Join(modelsPath, fileName), nil
}

func (s ModelEvalScripts) ScriptName(script ModelEvalScript) (string, error) {
	switch script {
	case ModelEvalSignalEvalScript:
		return s.SignalEval, nil
	case ModelEvalMMLUProVLLMEval:
		return s.MMLUProVLLMEval, nil
	case ModelEvalResultToConfigScript:
		return s.ResultToConfig, nil
	default:
		return "", fmt.Errorf("unknown model-eval script %q", script)
	}
}

func (m MLPipelineModelFiles) FileName(algorithm string) (string, error) {
	switch algorithm {
	case "knn":
		return m.KNN, nil
	case "kmeans":
		return m.KMeans, nil
	case "svm":
		return m.SVM, nil
	case "mlp":
		return m.MLP, nil
	default:
		return "", fmt.Errorf("unknown ml pipeline algorithm %q", algorithm)
	}
}

func (c Contract) RuntimeSelectionService(service RuntimeSelectionService) (TrainingServiceContract, error) {
	switch service {
	case RuntimeSelectionAutoMixVerifier:
		return c.RuntimeSelection.Services.AutoMixVerifier, nil
	case RuntimeSelectionRouterR1Server:
		return c.RuntimeSelection.Services.RouterR1Server, nil
	default:
		return TrainingServiceContract{}, fmt.Errorf("unknown runtime selection service %q", service)
	}
}

func (c Contract) HasModelEvalProjectLayout(projectRoot string) bool {
	required := []string{
		filepath.Join(c.ModelEval.RepoRelativeDir, c.ModelEval.Scripts.SignalEval),
		filepath.Join(c.ModelEval.RepoRelativeDir, c.ModelEval.Scripts.MMLUProVLLMEval),
	}
	return hasAllRepoRelativePaths(projectRoot, required)
}

func (c Contract) HasMLPipelineProjectLayout(projectRoot string) bool {
	required := []string{
		filepath.Join(c.MLPipeline.RepoRelativeDir, c.MLPipeline.Scripts.Benchmark),
	}
	return hasAllRepoRelativePaths(projectRoot, required)
}

func hasAllRepoRelativePaths(projectRoot string, repoRelativePaths []string) bool {
	for _, relPath := range repoRelativePaths {
		if _, err := os.Stat(contractJoin(projectRoot, relPath)); err != nil {
			return false
		}
	}
	return true
}

func FindProjectRootWithModelEval(candidates ...string) string {
	return currentContract.findProjectRoot(candidates, currentContract.HasModelEvalProjectLayout)
}

func FindProjectRootWithMLPipeline(candidates ...string) string {
	return currentContract.findProjectRoot(candidates, currentContract.HasMLPipelineProjectLayout)
}

func (c Contract) findProjectRoot(candidates []string, predicate func(string) bool) string {
	for _, candidate := range candidates {
		if root := findProjectRootFromCandidate(candidate, predicate); root != "" {
			return root
		}
	}
	return ""
}

func findProjectRootFromCandidate(start string, predicate func(string) bool) string {
	if start == "" {
		return ""
	}

	info, err := os.Stat(start)
	if err != nil {
		return ""
	}

	dir := filepath.Clean(start)
	if !info.IsDir() {
		dir = filepath.Dir(dir)
	}

	for {
		if predicate(dir) {
			return dir
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
