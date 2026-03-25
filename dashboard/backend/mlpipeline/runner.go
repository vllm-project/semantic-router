package mlpipeline

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
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
	wf           *workflowstore.Store
	dataDir      string // directory for uploads and outputs
	trainingDir  string // path to src/training/model_selection/ml_model_selection
	pythonPath   string
	mlServiceURL string // URL of the Python ML service (sidecar); empty = use subprocess
	progressChan chan ProgressUpdate
}

// RunnerConfig holds configuration for the Runner.
type RunnerConfig struct {
	DataDir      string
	TrainingDir  string
	PythonPath   string
	MLServiceURL string               // e.g. "http://ml-service:8686" (empty = subprocess mode)
	Workflow     *workflowstore.Store // required; durable job and progress state
}

// NewRunner creates a new ML onboarding runner.
func NewRunner(cfg RunnerConfig) (*Runner, error) {
	if cfg.Workflow == nil {
		return nil, fmt.Errorf("mlpipeline: Workflow store is required")
	}
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
		wf:           cfg.Workflow,
		dataDir:      cfg.DataDir,
		trainingDir:  cfg.TrainingDir,
		pythonPath:   cfg.PythonPath,
		mlServiceURL: strings.TrimRight(cfg.MLServiceURL, "/"),
		progressChan: make(chan ProgressUpdate, 100),
	}, nil
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
	_ = r.wf.UpdateMLJobProgress(jobID, percent, step)
	_ = r.wf.AppendMLProgressEvent(jobID, step, percent, message)
}

func (r *Runner) persistJob(j *Job) error {
	rec := jobToRecord(j)
	return r.wf.PutMLJob(rec)
}

func (r *Runner) setJobRunning(j *Job) {
	j.Status = StatusRunning
	_ = r.persistJob(j)
}

func (r *Runner) createJob(jobType string) *Job {
	id := fmt.Sprintf("ml-%s-%d", jobType, time.Now().UnixMilli())
	job := &Job{
		ID:        id,
		Type:      jobType,
		Status:    StatusPending,
		CreatedAt: time.Now(),
	}
	_ = r.persistJob(job)
	return job
}

// GetJob returns a job by ID.
func (r *Runner) GetJob(id string) *Job {
	rec, err := r.wf.GetMLJob(id)
	if err != nil || rec == nil {
		return nil
	}
	j := recordToJob(rec)
	return j
}

// ListJobs returns all jobs.
func (r *Runner) ListJobs() []*Job {
	recs, err := r.wf.ListMLJobs()
	if err != nil {
		return []*Job{}
	}
	result := make([]*Job, 0, len(recs))
	for i := range recs {
		j := recordToJob(&recs[i])
		result = append(result, j)
	}
	return result
}

func (r *Runner) failJob(jobID, errMsg string) {
	rec, err := r.wf.GetMLJob(jobID)
	if err != nil || rec == nil {
		return
	}
	rec.Status = string(StatusFailed)
	rec.Error = errMsg
	rec.CompletedAt = time.Now()
	_ = r.wf.PutMLJob(*rec)
}

func (r *Runner) completeJob(jobID string, outputFiles []string) {
	rec, err := r.wf.GetMLJob(jobID)
	if err != nil || rec == nil {
		return
	}
	rec.Status = string(StatusCompleted)
	rec.OutputFiles = outputFiles
	rec.CompletedAt = time.Now()
	rec.Progress = 100
	_ = r.wf.PutMLJob(*rec)
}

func jobToRecord(j *Job) workflowstore.MLJobRecord {
	return workflowstore.MLJobRecord{
		ID:          j.ID,
		Type:        j.Type,
		Status:      string(j.Status),
		CreatedAt:   j.CreatedAt,
		CompletedAt: j.CompletedAt,
		Error:       j.Error,
		OutputFiles: j.OutputFiles,
		Progress:    j.Progress,
		CurrentStep: j.CurrentStep,
	}
}

func recordToJob(rec *workflowstore.MLJobRecord) *Job {
	return &Job{
		ID:          rec.ID,
		Type:        rec.Type,
		Status:      JobStatus(rec.Status),
		CreatedAt:   rec.CreatedAt,
		CompletedAt: rec.CompletedAt,
		Error:       rec.Error,
		OutputFiles: rec.OutputFiles,
		Progress:    rec.Progress,
		CurrentStep: rec.CurrentStep,
	}
}

// ListProgressEvents returns durable typed progress records for a job.
func (r *Runner) ListProgressEvents(jobID string, limit int) ([]workflowstore.MLProgressEvent, error) {
	return r.wf.ListMLProgressEvents(jobID, limit)
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
