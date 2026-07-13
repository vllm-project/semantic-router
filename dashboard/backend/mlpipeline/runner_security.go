package mlpipeline

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
	"gopkg.in/yaml.v3"
)

const (
	maxManagedPathBytes       = 4096
	maxConfigDecisions        = 512
	maxDecisionValues         = 256
	maxDecisionStringRunes    = 256
	maxModelReferenceRunes    = 2048
	maxTrainingAlgorithms     = 4
	maxMLPHiddenLayers        = 4
	maxMLPHiddenSize          = 4096
	maxMLPHiddenUnits         = 8192
	maxMLPHiddenPairProducts  = 25_000_000
	maxMLPHiddenSizesRunes    = 128
	maxEmbeddingModelRunes    = 64
	managedUploadDirectory    = ".uploads"
	maximumManagedOutputFiles = 32
	maxBenchmarkModelsFile    = 4 << 20
	maxBenchmarkQueriesFile   = 64 << 20
	maxTrainingDataFile       = 64 << 20
	maxBenchmarkQueryLine     = 1 << 20
	maxBenchmarkQueries       = 20_000
	maxBenchmarkModels        = 64
	maxBenchmarkTasks         = 20_000
	maxBenchmarkTokenBudget   = 20_000_000
	maxTrainingRecords        = 200_000
	maxPipelineArtifactFile   = 256 << 20
	maxPipelineArtifactTotal  = 512 << 20
	maxGeneratedConfigFile    = 8 << 20
)

var (
	allowedAlgorithms = map[string]struct{}{
		"knn": {}, "kmeans": {}, "svm": {}, "mlp": {},
	}
	allowedDevices = map[string]struct{}{
		"cpu": {}, "cuda": {}, "mps": {},
	}
	allowedEmbeddingModels = map[string]struct{}{
		"qwen3": {}, "gte": {}, "mpnet": {}, "e5": {}, "bge": {},
	}
)

// ValidateBenchmarkRequest bounds work amplification before a benchmark job is
// admitted. Zero values retain their documented default semantics.
func ValidateBenchmarkRequest(req BenchmarkRequest) error {
	if req.Concurrency < 0 || req.Concurrency > 16 {
		return errors.New("concurrency must be between 0 and 16")
	}
	if req.MaxTokens < 0 || req.MaxTokens > 8192 {
		return errors.New("max_tokens must be between 0 and 8192")
	}
	if math.IsNaN(req.Temperature) || math.IsInf(req.Temperature, 0) || req.Temperature < 0 || req.Temperature > 2 {
		return errors.New("temperature must be between 0 and 2")
	}
	if req.Limit < 0 || req.Limit > maxBenchmarkQueries {
		return errors.New("limit must be between 0 and 20000")
	}
	return nil
}

// ValidateTrainRequest validates all values that cross the Go-to-Python
// process/service boundary. This prevents unbounded jobs and keeps strings from
// being reinterpreted as command-line options or model identifiers.
func ValidateTrainRequest(req TrainRequest) error {
	if len(req.Algorithms) > maxTrainingAlgorithms {
		return errors.New("too many algorithms")
	}
	seenAlgorithms := make(map[string]struct{}, len(req.Algorithms))
	for _, algorithm := range req.Algorithms {
		if _, ok := allowedAlgorithms[algorithm]; !ok {
			return errors.New("unsupported algorithm")
		}
		if _, duplicate := seenAlgorithms[algorithm]; duplicate {
			return errors.New("duplicate algorithm")
		}
		seenAlgorithms[algorithm] = struct{}{}
	}
	if req.Device != "" {
		if _, ok := allowedDevices[req.Device]; !ok {
			return errors.New("unsupported device")
		}
	}
	if req.EmbeddingModel != "" {
		if utf8.RuneCountInString(req.EmbeddingModel) > maxEmbeddingModelRunes {
			return errors.New("embedding model is too long")
		}
		if _, ok := allowedEmbeddingModels[req.EmbeddingModel]; !ok {
			return errors.New("unsupported embedding model")
		}
	}
	if !finiteInRange(req.QualityWeight, 0, 1) {
		return errors.New("quality_weight must be between 0 and 1")
	}
	if req.BatchSize < 0 || req.BatchSize > 1024 {
		return errors.New("batch_size must be between 0 and 1024")
	}
	if req.KnnK < 0 || req.KnnK > 1024 {
		return errors.New("knn_k must be between 0 and 1024")
	}
	if req.KmeansClusters < 0 || req.KmeansClusters > 4096 {
		return errors.New("kmeans_clusters must be between 0 and 4096")
	}
	if req.SvmKernel != "" && req.SvmKernel != "rbf" && req.SvmKernel != "linear" {
		return errors.New("unsupported svm_kernel")
	}
	if !finiteInRange(req.SvmGamma, 0, 1_000_000) {
		return errors.New("svm_gamma must be between 0 and 1000000")
	}
	if err := validateHiddenSizes(req.MlpHiddenSizes); err != nil {
		return err
	}
	if req.MlpEpochs < 0 || req.MlpEpochs > 500 {
		return errors.New("mlp_epochs must be between 0 and 500")
	}
	if !finiteInRange(req.MlpLearningRate, 0, 1) {
		return errors.New("mlp_learning_rate must be between 0 and 1")
	}
	if !finiteInRange(req.MlpDropout, 0, 1) {
		return errors.New("mlp_dropout must be between 0 and 1")
	}
	return nil
}

// ValidateConfigRequest bounds the generated YAML and rejects control
// characters at the API boundary. ModelsPath remains configurable because it
// is a deployment path, not a server-side file read.
func ValidateConfigRequest(req ConfigRequest) error {
	if len(req.ModelsPath) > maxManagedPathBytes || !safeControlPlaneString(req.ModelsPath, maxModelReferenceRunes, true) {
		return errors.New("models_path is invalid")
	}
	if req.Device != "" {
		if _, ok := allowedDevices[req.Device]; !ok {
			return errors.New("unsupported device")
		}
	}
	if len(req.Decisions) > maxConfigDecisions {
		return errors.New("too many decisions")
	}
	for _, decision := range req.Decisions {
		if !safeControlPlaneString(decision.Name, maxDecisionStringRunes, false) {
			return errors.New("decision name is invalid")
		}
		if _, ok := allowedAlgorithms[decision.Algorithm]; !ok {
			return errors.New("unsupported decision algorithm")
		}
		if len(decision.Domains) > maxDecisionValues || len(decision.ModelNames) > maxDecisionValues {
			return errors.New("decision has too many values")
		}
		for _, domain := range decision.Domains {
			if !safeControlPlaneString(domain, maxDecisionStringRunes, false) {
				return errors.New("decision domain is invalid")
			}
		}
		for _, modelName := range decision.ModelNames {
			if !safeControlPlaneString(modelName, maxDecisionStringRunes, false) {
				return errors.New("decision model name is invalid")
			}
		}
	}
	if len(req.ModelRefs) > maxConfigDecisions {
		return errors.New("too many model references")
	}
	for name, endpoint := range req.ModelRefs {
		if !safeControlPlaneString(name, maxDecisionStringRunes, false) ||
			!safeControlPlaneString(endpoint, maxModelReferenceRunes, false) {
			return errors.New("model reference is invalid")
		}
	}
	return nil
}

func finiteInRange(value, minimum, maximum float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= minimum && value <= maximum
}

func validateHiddenSizes(value string) error {
	if value == "" {
		return nil
	}
	if utf8.RuneCountInString(value) > maxMLPHiddenSizesRunes {
		return errors.New("mlp_hidden_sizes is too long")
	}
	parts := strings.Split(value, ",")
	if len(parts) == 0 || len(parts) > maxMLPHiddenLayers {
		return errors.New("mlp_hidden_sizes has too many layers")
	}
	totalUnits := 0
	pairProducts := 0
	previous := 1024 // embedding dimension used by the generated ML config
	for _, part := range parts {
		size, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || size <= 0 || size > maxMLPHiddenSize {
			return errors.New("mlp_hidden_sizes is invalid")
		}
		totalUnits += size
		pairProducts += previous * size
		previous = size
	}
	if totalUnits > maxMLPHiddenUnits || pairProducts > maxMLPHiddenPairProducts {
		return errors.New("mlp_hidden_sizes exceeds the training work budget")
	}
	return nil
}

type benchmarkModelsDocument struct {
	Models []struct {
		Name      string `yaml:"name"`
		MaxTokens int    `yaml:"max_tokens"`
	} `yaml:"models"`
}

func validateBenchmarkWorkload(modelsPath, queriesPath string, req BenchmarkRequest) error {
	modelsInfo, err := os.Stat(modelsPath)
	if err != nil || !modelsInfo.Mode().IsRegular() || modelsInfo.Size() > maxBenchmarkModelsFile {
		return errors.New("models YAML exceeds its work budget")
	}
	modelsRaw, err := os.ReadFile(modelsPath)
	if err != nil {
		return errors.New("models YAML could not be read")
	}
	var document benchmarkModelsDocument
	decoder := yaml.NewDecoder(bytes.NewReader(modelsRaw))
	if decodeErr := decoder.Decode(&document); decodeErr != nil || len(document.Models) == 0 || len(document.Models) > maxBenchmarkModels {
		return errors.New("models YAML has an invalid model count")
	}
	var trailingDocument any
	if trailingErr := decoder.Decode(&trailingDocument); !errors.Is(trailingErr, io.EOF) {
		return errors.New("models YAML must contain exactly one document")
	}
	tokensPerQuery := int64(0)
	for _, model := range document.Models {
		if !safeControlPlaneString(model.Name, maxDecisionStringRunes, false) {
			return errors.New("models YAML has an invalid model name")
		}
		modelTokens := model.MaxTokens
		if modelTokens == 0 {
			modelTokens = 1024
		}
		if modelTokens < 0 || modelTokens > 8192 {
			return errors.New("models YAML max_tokens exceeds its work budget")
		}
		tokensPerQuery += int64(modelTokens)
	}

	queriesInfo, err := os.Stat(queriesPath)
	if err != nil || !queriesInfo.Mode().IsRegular() || queriesInfo.Size() > maxBenchmarkQueriesFile {
		return errors.New("queries JSONL exceeds its work budget")
	}
	queriesFile, err := os.Open(queriesPath)
	if err != nil {
		return errors.New("queries JSONL could not be read")
	}
	defer queriesFile.Close()
	scanner := bufio.NewScanner(queriesFile)
	scanner.Buffer(make([]byte, 4096), maxBenchmarkQueryLine)
	queryCount := 0
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if len(line) > maxBenchmarkQueryLine || len(line) == 0 || line[0] != '{' || !json.Valid(line) {
			return errors.New("queries JSONL contains an invalid record")
		}
		queryCount++
		if queryCount > maxBenchmarkQueries {
			return errors.New("queries JSONL exceeds its record budget")
		}
	}
	if scanner.Err() != nil || queryCount == 0 {
		return errors.New("queries JSONL is invalid")
	}
	effectiveQueries := queryCount
	if req.Limit > 0 && req.Limit < effectiveQueries {
		effectiveQueries = req.Limit
	}
	if int64(effectiveQueries)*int64(len(document.Models)) > maxBenchmarkTasks ||
		int64(effectiveQueries)*tokensPerQuery > maxBenchmarkTokenBudget {
		return errors.New("benchmark request exceeds the combined work budget")
	}
	return nil
}

func validateTrainingDataFile(path string) error {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > maxTrainingDataFile {
		return errors.New("training data exceeds its file budget")
	}
	file, err := os.Open(path)
	if err != nil {
		return errors.New("training data could not be read")
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), maxBenchmarkQueryLine)
	records := 0
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if line[0] != '{' || !json.Valid(line) {
			return errors.New("training data contains an invalid record")
		}
		records++
		if records > maxTrainingRecords {
			return errors.New("training data exceeds its record budget")
		}
	}
	if scanner.Err() != nil || records == 0 {
		return errors.New("training data is invalid")
	}
	return nil
}

func safeControlPlaneString(value string, maxRunes int, allowEmpty bool) bool {
	if value == "" {
		return allowEmpty
	}
	if !utf8.ValidString(value) || strings.TrimSpace(value) != value || !norm.NFC.IsNormalString(value) || utf8.RuneCountInString(value) > maxRunes {
		return false
	}
	for _, r := range value {
		if r == 0 || unicode.IsControl(r) {
			return false
		}
	}
	return true
}

func validateMLServiceURL(rawURL string) (string, error) {
	if len(rawURL) > maxManagedPathBytes || !safeControlPlaneString(rawURL, maxManagedPathBytes, false) {
		return "", errors.New("ML service URL is invalid")
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Opaque != "" || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("ML service URL must be an absolute HTTP(S) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("ML service URL must not contain credentials, query parameters, or fragments")
	}
	return strings.TrimRight(parsed.String(), "/"), nil
}

// CreateUploadDir creates a server-named, private staging directory inside the
// runner data root. The caller owns it until a successful Run* handoff.
func (r *Runner) CreateUploadDir(prefix string) (string, error) {
	uploadRoot := filepath.Join(r.dataDir, managedUploadDirectory)
	if err := ensurePrivateDir(uploadRoot); err != nil {
		return "", fmt.Errorf("create upload root: %w", err)
	}
	dir, err := os.MkdirTemp(uploadRoot, prefix)
	if err != nil {
		return "", fmt.Errorf("create upload directory: %w", err)
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		_ = os.RemoveAll(dir)
		return "", fmt.Errorf("secure upload directory: %w", err)
	}
	return dir, nil
}

// RemoveUploadDir removes only a direct child created below the managed upload
// root. It will never follow an attacker-controlled path outside dataDir.
func (r *Runner) RemoveUploadDir(dir string) error {
	uploadRoot := filepath.Join(r.dataDir, managedUploadDirectory)
	cleanDir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(uploadRoot, cleanDir)
	if err != nil || rel == "." || filepath.IsAbs(rel) || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || strings.Contains(rel, string(filepath.Separator)) {
		return errors.New("upload directory is outside the managed upload root")
	}
	info, err := os.Lstat(cleanDir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("managed upload path is not a directory")
	}
	return os.RemoveAll(cleanDir)
}

func (r *Runner) cleanupManagedUploads(paths ...string) {
	seen := make(map[string]struct{}, len(paths))
	for _, path := range paths {
		parent := filepath.Dir(path)
		if _, exists := seen[parent]; exists {
			continue
		}
		seen[parent] = struct{}{}
		if err := r.RemoveUploadDir(parent); err != nil && !errors.Is(err, os.ErrNotExist) {
			// Paths not staged under .uploads are expected for prior job outputs.
			continue
		}
	}
}

// ValidateManagedFile resolves and validates an existing regular file beneath
// dataDir. It rejects leaf symlinks and canonical paths that escape the root.
func (r *Runner) ValidateManagedFile(path string) (string, error) {
	if path == "" || len(path) > maxManagedPathBytes {
		return "", errors.New("managed file path is invalid")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(r.dataDir, path)
	}
	cleanPath, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		return "", errors.New("managed file path is invalid")
	}
	leafInfo, err := os.Lstat(cleanPath)
	if err != nil {
		return "", fmt.Errorf("managed file unavailable: %w", err)
	}
	if leafInfo.Mode()&os.ModeSymlink != 0 || !leafInfo.Mode().IsRegular() {
		return "", errors.New("managed file must be a regular non-symlink file")
	}
	canonicalPath, err := filepath.EvalSymlinks(cleanPath)
	if err != nil {
		return "", fmt.Errorf("resolve managed file: %w", err)
	}
	if !pathWithinRoot(r.dataDir, canonicalPath) {
		return "", errors.New("managed file escapes the runner data directory")
	}
	return canonicalPath, nil
}

// OpenManagedFile validates a job artifact and returns an already-open file so
// HTTP serving does not re-resolve an attacker-swappable pathname.
func (r *Runner) OpenManagedFile(path string) (*os.File, os.FileInfo, error) {
	canonicalPath, err := r.ValidateManagedFile(path)
	if err != nil {
		return nil, nil, err
	}
	before, err := os.Stat(canonicalPath)
	if err != nil || !before.Mode().IsRegular() {
		return nil, nil, errors.New("managed file is unavailable")
	}
	file, err := os.Open(canonicalPath)
	if err != nil {
		return nil, nil, errors.New("managed file is unavailable")
	}
	after, err := file.Stat()
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		_ = file.Close()
		return nil, nil, errors.New("managed file changed while opening")
	}
	return file, after, nil
}

// ValidateJobOutputFile revalidates a persisted artifact against both dataDir
// and the producing job's narrower output directory.
func (r *Runner) ValidateJobOutputFile(jobID, jobType, path string) (string, error) {
	validated, err := r.validateOutputFiles(jobID, jobType, []string{path})
	if err != nil {
		return "", err
	}
	return validated[0], nil
}

// OpenJobOutputFile keeps the job-bound validation adjacent to opening the
// descriptor used for download.
func (r *Runner) OpenJobOutputFile(jobID, jobType, path string) (*os.File, os.FileInfo, error) {
	validated, err := r.ValidateJobOutputFile(jobID, jobType, path)
	if err != nil {
		return nil, nil, err
	}
	return r.OpenManagedFile(validated)
}

func (r *Runner) validateOutputFiles(jobID, jobType string, paths []string) ([]string, error) {
	if jobType != "benchmark" && jobType != "config" && jobType != "train" {
		return nil, errors.New("job has an invalid output class")
	}
	return r.validateOutputFilesInRoot(jobType, r.JobDir(jobID), paths)
}

// validateOutputFilesInRoot validates files against a server-selected output
// root. Completed job records always use JobDir(jobID); TrainDir is accepted
// only as the source root while immutable training snapshots are created.
func (r *Runner) validateOutputFilesInRoot(jobType, allowedRoot string, paths []string) ([]string, error) {
	if len(paths) == 0 || len(paths) > maximumManagedOutputFiles {
		return nil, errors.New("job produced an invalid number of output files")
	}
	allowedInfo, err := os.Lstat(allowedRoot)
	if err != nil || allowedInfo.Mode()&os.ModeSymlink != 0 || !allowedInfo.IsDir() {
		return nil, errors.New("job output directory is invalid")
	}
	canonicalAllowedRoot, err := filepath.EvalSymlinks(allowedRoot)
	if err != nil || !pathWithinRoot(r.dataDir, canonicalAllowedRoot) {
		return nil, errors.New("job output directory is invalid")
	}
	validated := make([]string, 0, len(paths))
	seen := make(map[string]struct{}, len(paths))
	totalBytes := int64(0)
	for _, path := range paths {
		canonicalPath, err := r.ValidateManagedFile(path)
		if err != nil {
			return nil, errors.New("job produced an invalid output file")
		}
		if !pathWithinRoot(canonicalAllowedRoot, canonicalPath) {
			return nil, errors.New("job output file is outside its job directory")
		}
		if filepath.Dir(canonicalPath) != canonicalAllowedRoot || !allowedOutputFilename(jobType, filepath.Base(canonicalPath)) {
			return nil, errors.New("job output filename is not allowed")
		}
		info, err := os.Stat(canonicalPath)
		if err != nil {
			return nil, errors.New("job output file is unavailable")
		}
		fileLimit := int64(maxPipelineArtifactFile)
		if jobType == "config" {
			fileLimit = maxGeneratedConfigFile
		}
		if info.Size() < 0 || info.Size() > fileLimit {
			return nil, errors.New("job output file exceeds its size budget")
		}
		totalBytes += info.Size()
		if totalBytes > maxPipelineArtifactTotal {
			return nil, errors.New("job output files exceed their total size budget")
		}
		if _, duplicate := seen[canonicalPath]; duplicate {
			return nil, errors.New("job produced duplicate output files")
		}
		seen[canonicalPath] = struct{}{}
		validated = append(validated, canonicalPath)
	}
	return validated, nil
}

func allowedOutputFilename(jobType, filename string) bool {
	switch jobType {
	case "benchmark":
		return filename == "benchmark_output.jsonl"
	case "config":
		return filename == "ml-model-selection-values.yaml"
	case "train":
		return filename == "knn_model.json" || filename == "kmeans_model.json" ||
			filename == "svm_model.json" || filename == "mlp_model.json"
	default:
		return false
	}
}

func pathWithinRoot(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !filepath.IsAbs(rel) && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func ensurePrivateDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("path is not a private directory")
	}
	return os.Chmod(dir, 0o700)
}

func cleanupStaleUploadDirs(dataDir string) error {
	uploadRoot := filepath.Join(dataDir, managedUploadDirectory)
	rootInfo, err := os.Lstat(uploadRoot)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if rootInfo.Mode()&os.ModeSymlink != 0 || !rootInfo.IsDir() {
		return errors.New("managed upload root is not a directory")
	}
	entries, err := os.ReadDir(uploadRoot)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		// os.RemoveAll removes a symlink itself and does not traverse its target.
		if err := os.RemoveAll(filepath.Join(uploadRoot, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}
