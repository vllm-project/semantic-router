package mlpipeline

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// JobStatus represents the status of a pipeline job.
type JobStatus string

const (
	StatusPending   JobStatus = "pending"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
)

// ProgressUpdate is sent to clients via SSE.
type ProgressUpdate struct {
	JobID   string `json:"job_id"`
	Step    string `json:"step"`
	Percent int    `json:"percent"`
	Message string `json:"message"`
}

// Job tracks a single pipeline execution.
type Job struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"` // "benchmark", "train", "config"
	Status      JobStatus `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Error       string    `json:"error,omitempty"`
	OutputFiles []string  `json:"output_files,omitempty"`
	Progress    int       `json:"progress"`
	CurrentStep string    `json:"current_step"`
}

// BenchmarkRequest is the input for Layer 1.
type BenchmarkRequest struct {
	Concurrency int     `json:"concurrency"`
	MaxTokens   int     `json:"max_tokens,omitempty"`  // default 1024
	Temperature float64 `json:"temperature,omitempty"` // default 0.0
	Concise     bool    `json:"concise,omitempty"`     // use concise prompts
	Limit       int     `json:"limit,omitempty"`       // limit number of queries (0 = no limit)
}

// TrainRequest is the input for Layer 2.
type TrainRequest struct {
	Algorithms     []string `json:"algorithms"`                // e.g. ["knn","kmeans","svm","mlp"]
	Device         string   `json:"device"`                    // "cpu", "cuda", "mps"
	EmbeddingModel string   `json:"embedding_model,omitempty"` // qwen3, gte, mpnet, e5, bge
	QualityWeight  float64  `json:"quality_weight,omitempty"`  // default 0.9
	BatchSize      int      `json:"batch_size,omitempty"`      // default 32
	// KNN params
	KnnK int `json:"knn_k,omitempty"` // default 5
	// KMeans params
	KmeansClusters int `json:"kmeans_clusters,omitempty"` // default 8
	// SVM params
	SvmKernel string  `json:"svm_kernel,omitempty"` // "rbf" or "linear"
	SvmGamma  float64 `json:"svm_gamma,omitempty"`  // default 1.0
	// MLP params
	MlpHiddenSizes  string  `json:"mlp_hidden_sizes,omitempty"`  // e.g. "256,128"
	MlpEpochs       int     `json:"mlp_epochs,omitempty"`        // default 100
	MlpLearningRate float64 `json:"mlp_learning_rate,omitempty"` // default 0.001
	MlpDropout      float64 `json:"mlp_dropout,omitempty"`       // default 0.1
}

// ConfigRequest is the input for Layer 3.
type ConfigRequest struct {
	ModelsPath string            `json:"models_path"`
	Device     string            `json:"device"` // cpu, cuda, mps — used for MLP config
	Decisions  []DecisionEntry   `json:"decisions"`
	ModelRefs  map[string]string `json:"model_refs,omitempty"` // optional: model name -> endpoint
}

// DecisionEntry defines a single routing decision for config generation.
type DecisionEntry struct {
	Name       string   `json:"name"`
	Domains    []string `json:"domains"`
	Algorithm  string   `json:"algorithm"`
	Priority   int      `json:"priority"`
	ModelNames []string `json:"model_names"`
}

// Runner orchestrates benchmark, train, and config generation steps.
type Runner struct {
	mu           sync.Mutex
	jobs         map[string]*Job
	dataDir      string // directory for uploads and outputs
	trainingDir  string // path to src/training/ml_model_selection
	pythonPath   string
	mlServiceURL string // URL of the Python ML service (sidecar); empty = use subprocess
	progressChan chan ProgressUpdate
}

// RunnerConfig holds configuration for the Runner.
type RunnerConfig struct {
	DataDir      string
	TrainingDir  string
	PythonPath   string
	MLServiceURL string // e.g. "http://ml-service:8686" (empty = subprocess mode)
}

// NewRunner creates a new ML onboarding runner.
func NewRunner(cfg RunnerConfig) *Runner {
	if cfg.PythonPath == "" {
		// On Windows, "python3" typically doesn't exist; use "python" instead
		if runtime.GOOS == "windows" {
			cfg.PythonPath = "python"
		} else {
			cfg.PythonPath = "python3"
		}
	}
	if cfg.DataDir == "" {
		cfg.DataDir = "./data/ml-pipeline"
	}
	// Make dataDir absolute so paths work correctly when cmd.Dir is set to trainingDir
	if abs, err := filepath.Abs(cfg.DataDir); err == nil {
		cfg.DataDir = abs
	}
	if err := ensureDir(cfg.DataDir); err != nil {
		log.Printf("Warning: could not create ML data dir %s: %v", cfg.DataDir, err)
	}
	if cfg.MLServiceURL != "" {
		log.Printf("[ml-pipeline] Using ML service at %s (HTTP mode)", cfg.MLServiceURL)
	} else {
		log.Printf("[ml-pipeline] No ML_SERVICE_URL set; using subprocess mode (python=%s)", cfg.PythonPath)
	}
	return &Runner{
		jobs:         make(map[string]*Job),
		dataDir:      cfg.DataDir,
		trainingDir:  cfg.TrainingDir,
		pythonPath:   cfg.PythonPath,
		mlServiceURL: strings.TrimRight(cfg.MLServiceURL, "/"),
		progressChan: make(chan ProgressUpdate, 100),
	}
}

// ProgressUpdates returns the channel for SSE streaming.
func (r *Runner) ProgressUpdates() <-chan ProgressUpdate {
	return r.progressChan
}

func (r *Runner) sendProgress(jobID string, percent int, step, message string) {
	update := ProgressUpdate{JobID: jobID, Step: step, Percent: percent, Message: message}
	select {
	case r.progressChan <- update:
	default:
	}
	r.mu.Lock()
	if j, ok := r.jobs[jobID]; ok {
		j.Progress = percent
		j.CurrentStep = step
	}
	r.mu.Unlock()
}

func (r *Runner) createJob(jobType string) *Job {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := fmt.Sprintf("ml-%s-%d", jobType, time.Now().UnixMilli())
	job := &Job{
		ID:        id,
		Type:      jobType,
		Status:    StatusPending,
		CreatedAt: time.Now(),
	}
	r.jobs[id] = job
	return job
}

// GetJob returns a job by ID.
func (r *Runner) GetJob(id string) *Job {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.jobs[id]
}

// ListJobs returns all jobs.
func (r *Runner) ListJobs() []*Job {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]*Job, 0, len(r.jobs))
	for _, j := range r.jobs {
		result = append(result, j)
	}
	return result
}

func (r *Runner) failJob(jobID, errMsg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if j, ok := r.jobs[jobID]; ok {
		j.Status = StatusFailed
		j.Error = errMsg
		j.CompletedAt = time.Now()
	}
}

func (r *Runner) completeJob(jobID string, outputFiles []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if j, ok := r.jobs[jobID]; ok {
		j.Status = StatusCompleted
		j.OutputFiles = outputFiles
		j.CompletedAt = time.Now()
		j.Progress = 100
	}
}

// JobDir returns the working directory for a given job.
func (r *Runner) JobDir(jobID string) string {
	return filepath.Join(r.dataDir, jobID)
}

// TrainDir returns the fixed directory for trained model output.
// All training runs write to the same directory so the path is stable.
func (r *Runner) TrainDir() string {
	return filepath.Join(r.dataDir, "ml-train")
}

// ensureDir creates a directory (and parents) if it does not exist.
// If the normal MkdirAll fails (e.g. WSL/NTFS ghost entries), it falls
// back to calling the system's "mkdir -p" command.
func ensureDir(dir string) error {
	// Fast path: already exists
	if info, err := os.Stat(dir); err == nil && info.IsDir() {
		return nil
	}
	// Normal creation
	if err := os.MkdirAll(dir, 0o755); err == nil {
		return nil
	}
	// Fallback: use system mkdir which may handle edge cases differently
	cmd := exec.Command("mkdir", "-p", dir)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("mkdir -p %s: %w (%s)", dir, err, string(out))
	}
	return nil
}

// scanCRLF is a bufio.SplitFunc that splits on \r or \n (or \r\n).
// This allows parsing tqdm progress bars that use \r to overwrite lines.
func scanCRLF(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	// Find the first \r or \n
	for i := 0; i < len(data); i++ {
		if data[i] == '\n' {
			return i + 1, data[0:i], nil
		}
		if data[i] == '\r' {
			// Check for \r\n
			if i+1 < len(data) && data[i+1] == '\n' {
				return i + 2, data[0:i], nil
			}
			return i + 1, data[0:i], nil
		}
	}
	// If at EOF, deliver the last token
	if atEOF {
		return len(data), data, nil
	}
	// Request more data
	return 0, nil, nil
}

// countFileLines counts non-empty lines in a file (for progress tracking).
func countFileLines(path string) int {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if len(strings.TrimSpace(scanner.Text())) > 0 {
			count++
		}
	}
	return count
}

// countYAMLModels counts the number of "- name:" entries in a YAML models file.
func countYAMLModels(path string) int {
	f, err := os.Open(path)
	if err != nil {
		return 1
	}
	defer f.Close()
	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "- name:") {
			count++
		}
	}
	if count == 0 {
		return 1
	}
	return count
}

// RunBenchmark runs Layer 1. If ML service URL is configured, it delegates to the
// Python HTTP sidecar. Otherwise, it spawns benchmark.py as a subprocess.
func (r *Runner) RunBenchmark(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) (string, error) {
	if r.mlServiceURL != "" {
		return r.runBenchmarkHTTP(ctx, modelsYAMLPath, queryJSONLPath, req)
	}
	return r.runBenchmarkSubprocess(ctx, modelsYAMLPath, queryJSONLPath, req)
}

// runBenchmarkSubprocess runs Layer 1: benchmark.py as a local subprocess.
func (r *Runner) runBenchmarkSubprocess(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) (string, error) {
	job := r.createJob("benchmark")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting benchmark", "Preparing benchmark run")

		outputFile := filepath.Join(jobDir, "benchmark_output.jsonl")
		concurrency := req.Concurrency
		if concurrency <= 0 {
			concurrency = 4
		}

		args := []string{
			filepath.Join(r.trainingDir, "benchmark.py"),
			"--queries", queryJSONLPath,
			"--model-config", modelsYAMLPath,
			"--output", outputFile,
			"--concurrency", fmt.Sprintf("%d", concurrency),
		}
		if req.MaxTokens > 0 {
			args = append(args, "--max-tokens", fmt.Sprintf("%d", req.MaxTokens))
		}
		if req.Temperature > 0 {
			args = append(args, "--temperature", fmt.Sprintf("%.2f", req.Temperature))
		}
		if req.Concise {
			args = append(args, "--concise")
		}
		if req.Limit > 0 {
			args = append(args, "--limit", fmt.Sprintf("%d", req.Limit))
		}

		r.sendProgress(job.ID, 10, "Running benchmark", fmt.Sprintf("Running benchmark.py with concurrency=%d", concurrency))

		cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath and args are server-controlled, not user input
		cmd.Dir = r.trainingDir
		// Keep tqdm ENABLED so we can parse real progress from its output.
		// We use a custom scanner that splits on \r (carriage return) to handle tqdm.
		cmd.Env = append(os.Environ(),
			"PYTHONIOENCODING=utf-8",
			"PYTHONUNBUFFERED=1",
			"HF_HUB_DISABLE_PROGRESS_BARS=1",
		)

		numQueries := countFileLines(queryJSONLPath)
		numModels := countYAMLModels(modelsYAMLPath)
		totalExpected := numQueries * numModels
		log.Printf("[benchmark/%s] Expecting ~%d results (%d queries × %d models)", job.ID, totalExpected, numQueries, numModels)

		// Merge stdout+stderr into a single pipe so we capture tqdm (stderr) + prints (stdout)
		pipeR, pipeW, err := os.Pipe()
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to create pipe: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		cmd.Stdout = pipeW
		cmd.Stderr = pipeW

		if err := cmd.Start(); err != nil {
			pipeW.Close()
			pipeR.Close()
			r.failJob(job.ID, fmt.Sprintf("failed to start benchmark: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		pipeW.Close() // Close write end; the child process holds its own copy

		// tqdm progress regex: matches "5/20" or "10/20" patterns
		tqdmProgressRe := regexp.MustCompile(`\b(\d+)/(\d+)\b`)

		// Read output in real-time with a custom scanner that splits on \r or \n.
		// This lets us parse tqdm's carriage-return-based progress updates.
		var outputBuf strings.Builder
		scanner := bufio.NewScanner(pipeR)
		scanner.Split(scanCRLF)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			outputBuf.WriteString(line + "\n")

			// Parse tqdm progress: look for "X/Y" pattern (e.g. "5/20")
			if matches := tqdmProgressRe.FindStringSubmatch(line); len(matches) == 3 {
				current, err1 := strconv.Atoi(matches[1])
				total, err2 := strconv.Atoi(matches[2])
				if err1 == nil && err2 == nil && total > 0 {
					// Map tqdm 0-100% to our 10-95% range
					tqdmPct := current * 100 / total
					pct := 10 + tqdmPct*85/100
					if pct > 95 {
						pct = 95
					}
					msg := fmt.Sprintf("Benchmarking: %d/%d queries completed (%d%%)", current, total, tqdmPct)
					r.sendProgress(job.ID, pct, "Running benchmark", msg)
				}
			}
		}
		pipeR.Close()

		waitErr := cmd.Wait()
		fullOutput := outputBuf.String()
		log.Printf("[benchmark/%s] Output:\n%s", job.ID, fullOutput)

		if waitErr != nil {
			r.failJob(job.ID, fmt.Sprintf("benchmark failed: %v", waitErr))
			r.sendProgress(job.ID, 100, "Failed", waitErr.Error())
			return
		}

		// Verify output exists
		if _, err := os.Stat(outputFile); err != nil {
			r.failJob(job.ID, "benchmark output file not created")
			r.sendProgress(job.ID, 100, "Failed", "Output file not found")
			return
		}

		r.completeJob(job.ID, []string{outputFile})
		r.sendProgress(job.ID, 100, "Completed", "Benchmark finished successfully")
	}()

	return job.ID, nil
}

// RunTrain runs Layer 2. If ML service URL is configured, it delegates to the
// Python HTTP sidecar. Otherwise, it spawns train.py as a subprocess.
func (r *Runner) RunTrain(ctx context.Context, benchmarkDataPath string, req TrainRequest) (string, error) {
	if r.mlServiceURL != "" {
		return r.runTrainHTTP(ctx, benchmarkDataPath, req)
	}
	return r.runTrainSubprocess(ctx, benchmarkDataPath, req)
}

// runTrainSubprocess runs Layer 2: train.py as a local subprocess.
// train.py supports: --data-file, --output-dir, --embedding-model, --device,
// --algorithm (all|knn|kmeans|svm|mlp), --skip-mlp, and MLP-specific args.
// If all 4 algorithms are requested, a single --algorithm all invocation is used.
// Otherwise, train.py is invoked once per algorithm (embeddings are cached between runs).
func (r *Runner) runTrainSubprocess(ctx context.Context, benchmarkDataPath string, req TrainRequest) (string, error) {
	job := r.createJob("train")
	// Use fixed ml-train directory so models always land in a stable path
	jobDir := r.TrainDir()
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting training", "Preparing training run")

		device := req.Device
		if device == "" {
			device = "cpu"
		}

		algorithms := req.Algorithms
		if len(algorithms) == 0 {
			algorithms = []string{"knn", "kmeans", "svm", "mlp"}
		}

		// Check if all 4 algorithms are selected → use single --algorithm all invocation
		allFour := len(algorithms) == 4
		if allFour {
			allSet := map[string]bool{}
			for _, a := range algorithms {
				allSet[a] = true
			}
			allFour = allSet["knn"] && allSet["kmeans"] && allSet["svm"] && allSet["mlp"]
		}

		// Build the list of train.py invocations: either ["all"] or individual algorithms
		var runs []string
		if allFour {
			runs = []string{"all"}
		} else {
			runs = algorithms
		}

		algNames := strings.Join(algorithms, ", ")
		r.sendProgress(job.ID, 10, "Running training", fmt.Sprintf("Training %s on device=%s", algNames, device))

		// Shared cache dir so embeddings are computed once and reused across runs
		cacheDir := filepath.Join(jobDir, ".cache")

		// Environment for all Python invocations
		pythonEnv := append(os.Environ(),
			"PYTHONIOENCODING=utf-8",
			"PYTHONUNBUFFERED=1",
			"TQDM_DISABLE=1",
			"HF_HUB_DISABLE_PROGRESS_BARS=1",
		)

		// Build dynamic progress stages based on selected algorithms
		stages := r.buildTrainProgressStages(algorithms)

		// Send periodic progress updates while training runs
		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(15 * time.Second)
			defer ticker.Stop()
			idx := 0
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					if idx < len(stages) {
						r.sendProgress(job.ID, stages[idx].pct, "Training", stages[idx].msg)
						idx++
					}
				}
			}
		}()

		// Build common args from request parameters
		commonArgs := []string{}
		if req.EmbeddingModel != "" {
			commonArgs = append(commonArgs, "--embedding-model", req.EmbeddingModel)
		}
		if req.QualityWeight > 0 {
			commonArgs = append(commonArgs, "--quality-weight", fmt.Sprintf("%.2f", req.QualityWeight))
		}
		if req.BatchSize > 0 {
			commonArgs = append(commonArgs, "--batch-size", fmt.Sprintf("%d", req.BatchSize))
		}
		if req.KnnK > 0 {
			commonArgs = append(commonArgs, "--knn-k", fmt.Sprintf("%d", req.KnnK))
		}
		if req.KmeansClusters > 0 {
			commonArgs = append(commonArgs, "--kmeans-clusters", fmt.Sprintf("%d", req.KmeansClusters))
		}
		if req.SvmKernel != "" {
			commonArgs = append(commonArgs, "--svm-kernel", req.SvmKernel)
		}
		if req.SvmGamma > 0 {
			commonArgs = append(commonArgs, "--svm-gamma", fmt.Sprintf("%.4f", req.SvmGamma))
		}
		if req.MlpHiddenSizes != "" {
			commonArgs = append(commonArgs, "--mlp-hidden-sizes", req.MlpHiddenSizes)
		}
		if req.MlpEpochs > 0 {
			commonArgs = append(commonArgs, "--mlp-epochs", fmt.Sprintf("%d", req.MlpEpochs))
		}
		if req.MlpLearningRate > 0 {
			commonArgs = append(commonArgs, "--mlp-learning-rate", fmt.Sprintf("%.6f", req.MlpLearningRate))
		}
		if req.MlpDropout > 0 {
			commonArgs = append(commonArgs, "--mlp-dropout", fmt.Sprintf("%.2f", req.MlpDropout))
		}

		// Execute train.py for each run (single "all" run, or one per algorithm)
		var lastErr error
		for _, algFlag := range runs {
			args := []string{
				filepath.Join(r.trainingDir, "train.py"),
				"--data-file", benchmarkDataPath,
				"--output-dir", jobDir,
				"--device", device,
				"--algorithm", algFlag,
				"--cache-dir", cacheDir,
			}
			args = append(args, commonArgs...)

			log.Printf("[train/%s] Running: %s %v", job.ID, r.pythonPath, args)

			cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath and args are server-controlled, not user input
			cmd.Dir = r.trainingDir
			cmd.Env = pythonEnv

			output, err := cmd.CombinedOutput()
			log.Printf("[train/%s] Output (algorithm=%s):\n%s", job.ID, algFlag, string(output))

			if err != nil {
				lastErr = fmt.Errorf("training %s failed: %w\n%s", algFlag, err, string(output))
				log.Printf("[train/%s] Error training %s: %v", job.ID, algFlag, err)
				// Continue with remaining algorithms instead of aborting
			}
		}

		close(done)

		if lastErr != nil && len(runs) == 1 {
			// Single algorithm/all failed — mark job as failed
			r.failJob(job.ID, lastErr.Error())
			r.sendProgress(job.ID, 100, "Failed", lastErr.Error())
			return
		}

		// Collect output model files
		outputFiles := []string{}
		for _, modelName := range []string{"knn_model.json", "kmeans_model.json", "svm_model.json", "mlp_model.json"} {
			modelFile := filepath.Join(jobDir, modelName)
			if _, err := os.Stat(modelFile); err == nil {
				outputFiles = append(outputFiles, modelFile)
			}
		}

		if len(outputFiles) == 0 {
			errMsg := "no model files were generated"
			if lastErr != nil {
				errMsg = lastErr.Error()
			}
			r.failJob(job.ID, errMsg)
			r.sendProgress(job.ID, 100, "Failed", errMsg)
			return
		}

		r.completeJob(job.ID, outputFiles)
		r.sendProgress(job.ID, 100, "Completed", fmt.Sprintf("Training finished: %d model(s) generated", len(outputFiles)))
	}()

	return job.ID, nil
}

// trainProgressStage defines a progress stage with percentage and message.
type trainProgressStage struct {
	pct int
	msg string
}

// buildTrainProgressStages builds dynamic progress stages based on selected algorithms.
func (r *Runner) buildTrainProgressStages(algorithms []string) []trainProgressStage {
	stages := []trainProgressStage{
		{15, "Loading data and embedding model..."},
		{25, "Generating query embeddings..."},
		{40, "Embedding generation in progress..."},
		{50, "Preparing training samples..."},
	}

	// Add per-algorithm stages, distributing from 55-90%
	algStages := []trainProgressStage{}
	for _, alg := range algorithms {
		switch alg {
		case "knn":
			algStages = append(algStages, trainProgressStage{0, "Training KNN model..."})
		case "kmeans":
			algStages = append(algStages, trainProgressStage{0, "Training KMeans model..."})
		case "svm":
			algStages = append(algStages, trainProgressStage{0, "Training SVM model..."})
		case "mlp":
			algStages = append(algStages, trainProgressStage{0, "Training MLP neural network..."})
		}
	}

	// Distribute algorithm stages evenly between 55-90%
	if len(algStages) > 0 {
		step := 35 / len(algStages) // 35% range (55-90)
		for i := range algStages {
			algStages[i].pct = 55 + i*step
		}
		stages = append(stages, algStages...)
	}

	stages = append(stages, trainProgressStage{90, "Finalizing models..."})
	return stages
}

// ---------------------------------------------------------------------------
// HTTP-based methods (production mode: calls the Python ML sidecar service)
// ---------------------------------------------------------------------------

// sseEvent represents a single SSE event from the ML service.
type sseEvent struct {
	Percent     int      `json:"percent"`
	Step        string   `json:"step"`
	Message     string   `json:"message"`
	Done        bool     `json:"done,omitempty"`
	Success     bool     `json:"success,omitempty"`
	OutputFiles []string `json:"output_files,omitempty"`
}

// readSSEStream reads SSE events from the ML service response and relays
// progress updates. It blocks until the stream is closed or a "done" event
// is received. Returns (outputFiles, error).
func (r *Runner) readSSEStream(jobID string, body io.ReadCloser) ([]string, error) {
	defer body.Close()

	scanner := bufio.NewScanner(body)
	// SSE lines can be long; increase buffer
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	var finalFiles []string
	var finalErr error

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {...json...}"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var evt sseEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			log.Printf("[ml-service/%s] Failed to parse SSE event: %v (data: %s)", jobID, err, data)
			continue
		}

		// Relay progress to our own SSE channel
		r.sendProgress(jobID, evt.Percent, evt.Step, evt.Message)

		if evt.Done {
			if evt.Success {
				finalFiles = evt.OutputFiles
			} else {
				finalErr = fmt.Errorf("ML service error: %s", evt.Message)
			}
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("SSE stream read error: %w", err)
	}

	return finalFiles, finalErr
}

// runBenchmarkHTTP delegates benchmark to the Python ML sidecar via HTTP.
func (r *Runner) runBenchmarkHTTP(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) (string, error) {
	job := r.createJob("benchmark")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting benchmark", "Sending request to ML service")

		// Build request body
		payload := map[string]interface{}{
			"queries_path":     queryJSONLPath,
			"models_yaml_path": modelsYAMLPath,
			"output_dir":       jobDir,
			"concurrency":      req.Concurrency,
			"max_tokens":       req.MaxTokens,
			"temperature":      req.Temperature,
			"concise":          req.Concise,
			"limit":            req.Limit,
		}

		body, err := json.Marshal(payload)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to marshal request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST",
			r.mlServiceURL+"/api/benchmark", bytes.NewReader(body))
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to create request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		log.Printf("[benchmark/%s] Calling ML service: POST %s/api/benchmark", job.ID, r.mlServiceURL)

		client := &http.Client{
			// No timeout — benchmark can take minutes
			Timeout: 0,
		}
		resp, err := client.Do(httpReq)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("ML service request failed: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			errMsg := fmt.Sprintf("ML service returned %d: %s", resp.StatusCode, string(respBody))
			r.failJob(job.ID, errMsg)
			r.sendProgress(job.ID, 100, "Failed", errMsg)
			return
		}

		// Read SSE stream
		outputFiles, streamErr := r.readSSEStream(job.ID, resp.Body)
		if streamErr != nil {
			r.failJob(job.ID, streamErr.Error())
			r.sendProgress(job.ID, 100, "Failed", streamErr.Error())
			return
		}

		r.completeJob(job.ID, outputFiles)
		r.sendProgress(job.ID, 100, "Completed", "Benchmark finished successfully")
	}()

	return job.ID, nil
}

// runTrainHTTP delegates training to the Python ML sidecar via HTTP.
func (r *Runner) runTrainHTTP(ctx context.Context, benchmarkDataPath string, req TrainRequest) (string, error) {
	job := r.createJob("train")
	// Use fixed ml-train directory so models always land in a stable path
	jobDir := r.TrainDir()
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting training", "Sending request to ML service")

		algorithms := req.Algorithms
		if len(algorithms) == 0 {
			algorithms = []string{"knn", "kmeans", "svm", "mlp"}
		}

		device := req.Device
		if device == "" {
			device = "cpu"
		}

		// Build request body matching the Python TrainRequest schema
		payload := map[string]interface{}{
			"data_file":         benchmarkDataPath,
			"output_dir":        jobDir,
			"algorithms":        algorithms,
			"device":            device,
			"embedding_model":   req.EmbeddingModel,
			"cache_dir":         filepath.Join(jobDir, ".cache"),
			"quality_weight":    req.QualityWeight,
			"batch_size":        req.BatchSize,
			"knn_k":             req.KnnK,
			"kmeans_clusters":   req.KmeansClusters,
			"svm_kernel":        req.SvmKernel,
			"svm_gamma":         req.SvmGamma,
			"mlp_hidden_sizes":  req.MlpHiddenSizes,
			"mlp_epochs":        req.MlpEpochs,
			"mlp_learning_rate": req.MlpLearningRate,
			"mlp_dropout":       req.MlpDropout,
		}

		body, err := json.Marshal(payload)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to marshal request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST",
			r.mlServiceURL+"/api/train", bytes.NewReader(body))
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to create request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		algNames := strings.Join(algorithms, ", ")
		log.Printf("[train/%s] Calling ML service: POST %s/api/train (algorithms=%s, device=%s)",
			job.ID, r.mlServiceURL, algNames, device)

		client := &http.Client{
			Timeout: 0, // Training can take many minutes
		}
		resp, err := client.Do(httpReq)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("ML service request failed: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			errMsg := fmt.Sprintf("ML service returned %d: %s", resp.StatusCode, string(respBody))
			r.failJob(job.ID, errMsg)
			r.sendProgress(job.ID, 100, "Failed", errMsg)
			return
		}

		// Read SSE stream
		outputFiles, streamErr := r.readSSEStream(job.ID, resp.Body)
		if streamErr != nil {
			r.failJob(job.ID, streamErr.Error())
			r.sendProgress(job.ID, 100, "Failed", streamErr.Error())
			return
		}

		r.completeJob(job.ID, outputFiles)
		r.sendProgress(job.ID, 100, "Completed",
			fmt.Sprintf("Training finished: %d model(s) generated", len(outputFiles)))
	}()

	return job.ID, nil
}

// GenerateConfig runs Layer 3: generates a deployment-ready YAML config.
func (r *Runner) GenerateConfig(req ConfigRequest) (string, error) {
	job := r.createJob("config")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	r.sendProgress(job.ID, 10, "Generating config", "Building deployment configuration")

	configMap := buildConfigYAML(req)

	outputPath := filepath.Join(jobDir, "ml-model-selection-values.yaml")
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(configMap); err != nil {
		r.failJob(job.ID, fmt.Sprintf("failed to marshal config: %v", err))
		return "", err
	}
	enc.Close()

	// Post-process: add blank line between decisions for readability
	yamlStr := buf.String()
	lines := strings.Split(yamlStr, "\n")
	var out []string
	decisionCount := 0
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimRight(line, " "), "    - name:") {
			decisionCount++
			// Add blank line before 2nd+ decision (not the first one)
			if decisionCount > 1 && len(out) > 0 && strings.TrimSpace(out[len(out)-1]) != "" {
				out = append(out, "")
			}
		}
		out = append(out, line)
	}
	finalYAML := strings.Join(out, "\n")

	if err := os.WriteFile(outputPath, []byte(finalYAML), 0o644); err != nil {
		r.failJob(job.ID, fmt.Sprintf("failed to write config: %v", err))
		return "", err
	}

	r.completeJob(job.ID, []string{outputPath})
	r.sendProgress(job.ID, 100, "Completed", "Config generated successfully")

	return job.ID, nil
}

// YAML config structs matching semantic-router values.yaml format.
type yamlConfig struct {
	Config yamlConfigInner `yaml:"config"`
}

type yamlConfigInner struct {
	ModelSelection yamlModelSelection `yaml:"model_selection"`
	Strategy       string             `yaml:"strategy"`
	Decisions      []yamlDecision     `yaml:"decisions"`
}

type yamlModelSelection struct {
	Enabled bool   `yaml:"enabled"`
	ML      yamlML `yaml:"ml"`
}

type yamlML struct {
	ModelsPath   string      `yaml:"models_path"`
	EmbeddingDim int         `yaml:"embedding_dim"`
	KNN          *yamlKNN    `yaml:"knn,omitempty"`
	KMeans       *yamlKMeans `yaml:"kmeans,omitempty"`
	SVM          *yamlSVM    `yaml:"svm,omitempty"`
	MLP          *yamlMLP    `yaml:"mlp,omitempty"`
}

type yamlKNN struct {
	K              int    `yaml:"k"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlKMeans struct {
	NumClusters    int    `yaml:"num_clusters"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlSVM struct {
	Kernel         string  `yaml:"kernel"`
	Gamma          float64 `yaml:"gamma"`
	PretrainedPath string  `yaml:"pretrained_path"`
}

type yamlMLP struct {
	Device         string `yaml:"device"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlDecision struct {
	Name      string         `yaml:"name"`
	Priority  int            `yaml:"priority"`
	Rules     yamlRules      `yaml:"rules"`
	Algorithm yamlAlgorithm  `yaml:"algorithm"`
	ModelRefs []yamlModelRef `yaml:"modelRefs"`
}

type yamlRules struct {
	Operator   string          `yaml:"operator"`
	Conditions []yamlCondition `yaml:"conditions"`
}

type yamlCondition struct {
	Type string `yaml:"type"`
	Name string `yaml:"name"`
}

type yamlAlgorithm struct {
	Type string `yaml:"type"`
}

type yamlModelRef struct {
	Model        string `yaml:"model"`
	UseReasoning bool   `yaml:"use_reasoning"`
}

// buildConfigYAML creates the config structure matching the semantic-router values.yaml format.
func buildConfigYAML(req ConfigRequest) yamlConfig {
	modelsPath := req.ModelsPath
	if modelsPath == "" {
		modelsPath = "/data/ml-pipeline/ml-train"
	}

	ml := yamlML{
		ModelsPath:   modelsPath,
		EmbeddingDim: 1024,
	}

	// Add per-algorithm config based on decisions
	algSet := map[string]bool{}
	for _, d := range req.Decisions {
		algSet[d.Algorithm] = true
	}
	if algSet["knn"] {
		ml.KNN = &yamlKNN{
			K:              5,
			PretrainedPath: filepath.Join(modelsPath, "knn_model.json"),
		}
	}
	if algSet["kmeans"] {
		ml.KMeans = &yamlKMeans{
			NumClusters:    8,
			PretrainedPath: filepath.Join(modelsPath, "kmeans_model.json"),
		}
	}
	if algSet["svm"] {
		ml.SVM = &yamlSVM{
			Kernel:         "rbf",
			Gamma:          1.0,
			PretrainedPath: filepath.Join(modelsPath, "svm_model.json"),
		}
	}
	if algSet["mlp"] {
		mlpDevice := req.Device
		if mlpDevice == "" {
			mlpDevice = "cpu"
		}
		ml.MLP = &yamlMLP{
			Device:         mlpDevice,
			PretrainedPath: filepath.Join(modelsPath, "mlp_model.json"),
		}
	}

	// Build decisions list
	decisions := []yamlDecision{}
	for _, d := range req.Decisions {
		conditions := []yamlCondition{}
		for _, dom := range d.Domains {
			conditions = append(conditions, yamlCondition{
				Type: "domain",
				Name: dom,
			})
		}

		modelRefs := []yamlModelRef{}
		for _, mn := range d.ModelNames {
			modelRefs = append(modelRefs, yamlModelRef{
				Model:        mn,
				UseReasoning: false,
			})
		}

		decisions = append(decisions, yamlDecision{
			Name:     d.Name,
			Priority: d.Priority,
			Rules: yamlRules{
				Operator:   "OR",
				Conditions: conditions,
			},
			Algorithm: yamlAlgorithm{Type: d.Algorithm},
			ModelRefs: modelRefs,
		})
	}

	return yamlConfig{
		Config: yamlConfigInner{
			ModelSelection: yamlModelSelection{
				Enabled: true,
				ML:      ml,
			},
			Strategy:  "priority",
			Decisions: decisions,
		},
	}
}
