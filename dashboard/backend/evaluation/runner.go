package evaluation

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

const evaluationAuthorizationEnv = "VLLM_SR_EVALUATION_BEARER_TOKEN"

// InferenceAuthorization is an in-memory, run-scoped credential. Its secret is
// deliberately unexported so it cannot become part of a task, result, or API
// response by accident.
type InferenceAuthorization struct {
	bearerToken string
}

// NewInferenceAuthorization validates a Router-issued delegated inference
// credential before it crosses into the task runner.
func NewInferenceAuthorization(bearerToken string) (InferenceAuthorization, error) {
	if strings.TrimSpace(bearerToken) == "" || strings.ContainsAny(bearerToken, "\r\n\t ") {
		return InferenceAuthorization{}, errors.New("evaluation inference credential is invalid")
	}
	return InferenceAuthorization{bearerToken: bearerToken}, nil
}

// Runner executes evaluation benchmarks.
type Runner struct {
	db              *DB
	projectRoot     string
	pythonPath      string
	resultsDir      string
	maxConcurrent   int
	activeProcesses sync.Map // map[taskID]*exec.Cmd
	progressChan    chan models.ProgressUpdate
}

// RunnerConfig holds configuration for the Runner.
type RunnerConfig struct {
	DB            *DB
	ProjectRoot   string
	PythonPath    string
	ResultsDir    string
	MaxConcurrent int
}

// NewRunner creates a new evaluation runner.
func NewRunner(cfg RunnerConfig) *Runner {
	if cfg.PythonPath == "" {
		cfg.PythonPath = "python3"
	}
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = 3
	}
	if cfg.ResultsDir == "" {
		cfg.ResultsDir = filepath.Join(cfg.ProjectRoot, "data", "results")
	} else if !filepath.IsAbs(cfg.ResultsDir) {
		// Make relative paths absolute based on project root
		cfg.ResultsDir = filepath.Join(cfg.ProjectRoot, cfg.ResultsDir)
	}

	// Ensure results directory exists
	if err := os.MkdirAll(cfg.ResultsDir, 0o755); err != nil {
		log.Printf("Warning: could not create results directory: %v", err)
	}

	log.Printf("Evaluation results directory: %s", cfg.ResultsDir)

	return &Runner{
		db:            cfg.DB,
		projectRoot:   cfg.ProjectRoot,
		pythonPath:    cfg.PythonPath,
		resultsDir:    cfg.ResultsDir,
		maxConcurrent: cfg.MaxConcurrent,
		progressChan:  make(chan models.ProgressUpdate, 100),
	}
}

// ProgressUpdates returns a channel for receiving progress updates.
func (r *Runner) ProgressUpdates() <-chan models.ProgressUpdate {
	return r.progressChan
}

// sendProgress sends a progress update.
func (r *Runner) sendProgress(taskID string, percent int, step, message string) {
	update := models.ProgressUpdate{
		TaskID:          taskID,
		ProgressPercent: percent,
		CurrentStep:     step,
		Message:         message,
		Timestamp:       time.Now().UnixMilli(),
	}

	// Non-blocking send
	select {
	case r.progressChan <- update:
	default:
		// Channel full, skip update
	}

	// Also update database
	if err := r.db.UpdateTaskProgress(taskID, percent, step); err != nil {
		log.Printf("Failed to update task progress in DB: %v", err)
	}
}

// RunTask executes an evaluation task.
func (r *Runner) RunTask(
	ctx context.Context,
	taskID string,
	authorization InferenceAuthorization,
) error {
	if authorization.bearerToken == "" {
		return errors.New("evaluation inference authorization is required")
	}
	task, err := r.db.GetTask(taskID)
	if err != nil {
		return fmt.Errorf("failed to get task: %w", err)
	}
	if task == nil {
		return fmt.Errorf("task not found: %s", taskID)
	}

	// Update status to running when the caller has not already transitioned the task.
	if task.Status != models.StatusRunning {
		if statusErr := r.db.UpdateTaskStatus(taskID, models.StatusRunning, ""); statusErr != nil {
			return fmt.Errorf("failed to update task status: %w", statusErr)
		}
	}

	r.sendProgress(taskID, 0, "Starting evaluation", "Initializing evaluation task")

	// Create task-specific output directory
	taskOutputDir := filepath.Join(r.resultsDir, taskID)
	if mkdirErr := os.MkdirAll(taskOutputDir, 0o755); mkdirErr != nil {
		_ = r.db.UpdateTaskStatus(taskID, models.StatusFailed, fmt.Sprintf("Failed to create output directory: %v", mkdirErr))
		return fmt.Errorf("failed to create output directory: %w", mkdirErr)
	}

	runErrors, err := r.runTaskDimensions(ctx, task, taskOutputDir, authorization)
	if err != nil {
		return err
	}

	if len(runErrors) > 0 {
		errorMessage := strings.Join(runErrors, "\n")
		if statusErr := r.db.UpdateTaskStatus(taskID, models.StatusFailed, errorMessage); statusErr != nil {
			return fmt.Errorf("failed to update task status after evaluation errors: %w", statusErr)
		}
		return fmt.Errorf("evaluation task failed:\n%s", errorMessage)
	}

	r.sendProgress(taskID, 100, "Completed", "All evaluations finished")
	if statusErr := r.db.UpdateTaskStatus(taskID, models.StatusCompleted, ""); statusErr != nil {
		return fmt.Errorf("failed to update task status: %w", statusErr)
	}

	return nil
}

func (r *Runner) runTaskDimensions(
	ctx context.Context,
	task *models.EvaluationTask,
	taskOutputDir string,
	authorization InferenceAuthorization,
) ([]string, error) {
	totalDimensions := len(task.Config.Dimensions)
	completedDimensions := 0
	var runErrors []string
	for _, dimension := range task.Config.Dimensions {
		select {
		case <-ctx.Done():
			_ = r.db.UpdateTaskStatus(task.ID, models.StatusCancelled, "Task cancelled")
			return nil, ctx.Err()
		default:
		}
		progressBase := (completedDimensions * 100) / totalDimensions
		step := fmt.Sprintf("Evaluating %s", dimension)
		r.sendProgress(task.ID, progressBase, step, fmt.Sprintf("Starting %s evaluation", dimension))
		datasets := task.Config.Datasets[string(dimension)]
		if len(datasets) == 0 {
			datasets = []string{getDefaultDataset(dimension)}
		}
		for _, dataset := range datasets {
			result, runErr := r.runDimensionDataset(
				ctx, task, dimension, dataset, taskOutputDir, authorization,
			)
			if runErr != nil {
				log.Printf("Error running %s evaluation on dataset %s: %v", dimension, dataset, runErr)
				runErrors = append(runErrors, fmt.Sprintf("%s/%s: %v", dimension, dataset, runErr))
				continue
			}
			if result != nil {
				if err := r.db.SaveResult(result); err != nil {
					log.Printf("Failed to save result: %v", err)
				}
				r.saveHistoricalMetrics(result)
			}
		}
		completedDimensions++
		progress := (completedDimensions * 100) / totalDimensions
		r.sendProgress(task.ID, progress, step, fmt.Sprintf("Completed %s evaluation", dimension))
	}
	return runErrors, nil
}

func (r *Runner) runDimensionDataset(
	ctx context.Context,
	task *models.EvaluationTask,
	dimension models.EvaluationDimension,
	dataset string,
	taskOutputDir string,
	authorization InferenceAuthorization,
) (*models.EvaluationResult, error) {
	switch dimension {
	case models.DimensionDomain, models.DimensionFactCheck, models.DimensionUserFeedback:
		return r.runSignalEvaluation(
			ctx, task.ID, task.Config, string(dimension), dataset, taskOutputDir, authorization,
		)
	case models.DimensionAccuracy:
		return r.runSystemEvaluation(ctx, task.ID, task.Config, dataset, taskOutputDir, authorization)
	default:
		log.Printf("Unknown dimension: %s", dimension)
		return nil, nil
	}
}

// CancelTask cancels a running evaluation task.
func (r *Runner) CancelTask(taskID string) error {
	if cmdVal, ok := r.activeProcesses.Load(taskID); ok {
		cmd := cmdVal.(*exec.Cmd)
		if cmd.Process != nil {
			if err := cmd.Process.Kill(); err != nil {
				log.Printf("Failed to kill process for task %s: %v", taskID, err)
			}
		}
		r.activeProcesses.Delete(taskID)
	}

	return r.db.UpdateTaskStatus(taskID, models.StatusCancelled, "Task cancelled by user")
}

// getDefaultDataset returns the default dataset ID for a given dimension.
func getDefaultDataset(dimension models.EvaluationDimension) string {
	switch dimension {
	case models.DimensionDomain:
		return "mmlu-pro-en"
	case models.DimensionFactCheck:
		return "fact-check-en"
	case models.DimensionUserFeedback:
		return "feedback-en"
	case models.DimensionAccuracy:
		return "mmlu-pro"
	default:
		return "default"
	}
}

// runSignalEvaluation runs the signal evaluation for a specific dataset.
func (r *Runner) runSignalEvaluation(
	ctx context.Context,
	taskID string,
	cfg models.EvaluationConfig,
	dimension string,
	datasetID string,
	outputDir string,
	authorization InferenceAuthorization,
) (*models.EvaluationResult, error) {
	outputPath := filepath.Join(outputDir, fmt.Sprintf("signal_eval_%s.json", datasetID))

	// Use endpoint as-is for eval API
	endpoint := strings.TrimSuffix(cfg.Endpoint, "/")

	// Build command arguments
	args := []string{
		r.modelEvalScriptPath("signal_eval.py"),
		"--dataset", datasetID,
		"--endpoint", endpoint,
		"--output", outputPath,
	}

	if cfg.MaxSamples > 0 {
		args = append(args, "--max_samples", fmt.Sprintf("%d", cfg.MaxSamples))
	}

	// Add concurrent parameter if specified
	if cfg.Concurrent > 0 {
		args = append(args, "--concurrent", fmt.Sprintf("%d", cfg.Concurrent))
	}

	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath is configured at startup, not user input
	cmd.Dir = r.projectRoot
	cmd.Env = r.pythonEnv(authorization)

	r.activeProcesses.Store(taskID, cmd)
	defer r.activeProcesses.Delete(taskID)

	_, err := r.runCommandWithProgress(ctx, cmd, taskID, datasetID)
	if err != nil {
		return nil, fmt.Errorf("signal evaluation failed: %w", err)
	}

	// Parse output JSON
	metrics, err := ParseSignalEvalOutput(outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse signal evaluation output: %w", err)
	}

	return &models.EvaluationResult{
		TaskID:         taskID,
		Dimension:      models.EvaluationDimension(dimension),
		DatasetName:    datasetID,
		Metrics:        metrics,
		RawResultsPath: outputPath,
	}, nil
}

// runSystemEvaluation runs system-level (MoM) evaluation, e.g. MMLU-Pro accuracy against an endpoint.
func (r *Runner) runSystemEvaluation(
	ctx context.Context,
	taskID string,
	cfg models.EvaluationConfig,
	datasetID string,
	outputDir string,
	authorization InferenceAuthorization,
) (*models.EvaluationResult, error) {
	// Only mmlu-pro is supported for accuracy dimension
	if datasetID != "mmlu-pro" {
		log.Printf("Unsupported system eval dataset: %s, using mmlu-pro", datasetID)
		datasetID = "mmlu-pro"
	}

	outDir := filepath.Join(outputDir, "system_eval_accuracy")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create output dir: %w", err)
	}

	endpoint := strings.TrimSuffix(cfg.Endpoint, "/")
	samplesPerCat := cfg.SamplesPerCat
	if samplesPerCat <= 0 {
		samplesPerCat = 5
	}
	args := []string{
		r.modelEvalScriptPath("mmlu_pro_vllm_eval.py"),
		"--endpoint", endpoint,
		"--output-dir", outDir,
		"--samples-per-category", fmt.Sprintf("%d", samplesPerCat),
	}
	if cfg.Concurrent > 0 {
		args = append(args, "--concurrent-requests", fmt.Sprintf("%d", cfg.Concurrent))
	}
	if cfg.Model != "" {
		args = append(args, "--models", cfg.Model)
	}

	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath is configured at startup
	cmd.Dir = r.projectRoot
	cmd.Env = r.pythonEnv(authorization)

	r.activeProcesses.Store(taskID, cmd)
	defer r.activeProcesses.Delete(taskID)

	_, err := r.runCommandWithProgress(ctx, cmd, taskID, "accuracy")
	if err != nil {
		return nil, fmt.Errorf("system evaluation failed: %w", err)
	}

	metrics, err := ParseMMLUProOutput(outDir)
	if err != nil {
		return nil, fmt.Errorf("failed to parse system evaluation output: %w", err)
	}

	return &models.EvaluationResult{
		TaskID:         taskID,
		Dimension:      models.DimensionAccuracy,
		DatasetName:    datasetID,
		Metrics:        metrics,
		RawResultsPath: outDir,
	}, nil
}

// runCommandWithProgress executes a command and captures output with progress updates.
func (r *Runner) runCommandWithProgress(ctx context.Context, cmd *exec.Cmd, taskID, dimension string) (string, error) {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return "", fmt.Errorf("failed to get stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start command: %w", err)
	}

	var output strings.Builder
	var errOutput strings.Builder

	// Read stdout
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			output.WriteString(line + "\n")
			if pct, ok := tqdmPercentFromLine(line); ok {
				r.sendProgress(taskID, pct, dimension, fmt.Sprintf("Processing: %d%%", pct))
			}
		}
	}()

	// Read stderr - tqdm writes progress here
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			errOutput.WriteString(line + "\n")
			if pct, ok := tqdmPercentFromLine(line); ok {
				r.sendProgress(taskID, pct, dimension, fmt.Sprintf("Processing: %d%%", pct))
			}
		}
	}()

	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", fmt.Errorf("command failed: %w\nstderr: %s", err, errOutput.String())
	}

	return output.String(), nil
}

func (r *Runner) modelEvalScriptPath(scriptName string) string {
	return filepath.Join(r.projectRoot, "src", "training", "model_eval", scriptName)
}

func (r *Runner) pythonEnv(authorization InferenceAuthorization) []string {
	pythonPath := r.projectRoot
	if existing := os.Getenv("PYTHONPATH"); existing != "" {
		pythonPath += string(os.PathListSeparator) + existing
	}

	environment := make([]string, 0, len(os.Environ())+2)
	for _, entry := range os.Environ() {
		if !strings.HasPrefix(entry, evaluationAuthorizationEnv+"=") &&
			!strings.HasPrefix(entry, "PYTHONPATH=") {
			environment = append(environment, entry)
		}
	}
	return append(
		environment,
		"PYTHONPATH="+pythonPath,
		evaluationAuthorizationEnv+"="+authorization.bearerToken,
	)
}

// saveHistoricalMetrics saves key metrics to the history table.
func (r *Runner) saveHistoricalMetrics(result *models.EvaluationResult) {
	// Define which metrics to track historically
	keyMetrics := []string{
		"precision", "recall", "f1_score", "accuracy",
		"avg_latency_ms", "p50_latency_ms", "p99_latency_ms",
		"efficiency_gain_percent",
	}

	for _, metricName := range keyMetrics {
		if value, ok := result.Metrics[metricName]; ok {
			var floatValue float64
			switch v := value.(type) {
			case float64:
				floatValue = v
			case int:
				floatValue = float64(v)
			case int64:
				floatValue = float64(v)
			default:
				continue
			}

			entry := &models.EvaluationHistoryEntry{
				ResultID:    result.ID,
				MetricName:  metricName,
				MetricValue: floatValue,
				RecordedAt:  time.Now(),
			}

			if err := r.db.SaveHistoryEntry(entry); err != nil {
				log.Printf("Failed to save history entry for %s: %v", metricName, err)
			}
		}
	}
}

// GetAvailableDatasets returns a list of available datasets grouped by dimension.
func GetAvailableDatasets() map[string][]models.DatasetInfo {
	return map[string][]models.DatasetInfo{
		string(models.DimensionDomain): {
			{
				Name:        "mmlu-pro-en",
				Description: "MMLU-Pro (English)",
				Dimension:   models.DimensionDomain,
				Level:       models.LevelRouter,
			},
			// MMLU-ProX multilingual datasets (29 languages)
			{Name: "mmlu-prox-zh", Description: "MMLU-ProX (Chinese)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-de", Description: "MMLU-ProX (German)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-en", Description: "MMLU-ProX (English)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-es", Description: "MMLU-ProX (Spanish)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-fr", Description: "MMLU-ProX (French)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-it", Description: "MMLU-ProX (Italian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ja", Description: "MMLU-ProX (Japanese)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ko", Description: "MMLU-ProX (Korean)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-af", Description: "MMLU-ProX (Afrikaans)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ar", Description: "MMLU-ProX (Arabic)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-bn", Description: "MMLU-ProX (Bengali)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-cs", Description: "MMLU-ProX (Czech)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-hi", Description: "MMLU-ProX (Hindi)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-hu", Description: "MMLU-ProX (Hungarian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-id", Description: "MMLU-ProX (Indonesian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-mr", Description: "MMLU-ProX (Marathi)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ne", Description: "MMLU-ProX (Nepali)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-pt", Description: "MMLU-ProX (Portuguese)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ru", Description: "MMLU-ProX (Russian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-sr", Description: "MMLU-ProX (Serbian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-sw", Description: "MMLU-ProX (Swahili)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-te", Description: "MMLU-ProX (Telugu)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-th", Description: "MMLU-ProX (Thai)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-uk", Description: "MMLU-ProX (Ukrainian)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-ur", Description: "MMLU-ProX (Urdu)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-vi", Description: "MMLU-ProX (Vietnamese)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-wo", Description: "MMLU-ProX (Wolof)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-yo", Description: "MMLU-ProX (Yoruba)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
			{Name: "mmlu-prox-zu", Description: "MMLU-ProX (Zulu)", Dimension: models.DimensionDomain, Level: models.LevelRouter},
		},
		string(models.DimensionFactCheck): {
			{
				Name:        "fact-check-en",
				Description: "Fact Check (English) - Binary classification",
				Dimension:   models.DimensionFactCheck,
				Level:       models.LevelRouter,
			},
		},
		string(models.DimensionUserFeedback): {
			{
				Name:        "feedback-en",
				Description: "User Feedback (English) - 4-class detection",
				Dimension:   models.DimensionUserFeedback,
				Level:       models.LevelRouter,
			},
		},
		string(models.DimensionAccuracy): {
			{
				Name:        "mmlu-pro",
				Description: "MMLU-Pro system accuracy via chat completions endpoint",
				Dimension:   models.DimensionAccuracy,
				Level:       models.LevelMoM,
			},
		},
	}
}

// ExportResults exports evaluation results in the specified format.
func (r *Runner) ExportResults(taskID string, format models.ExportFormat) ([]byte, string, error) {
	results, err := r.db.GetResults(taskID)
	if err != nil {
		return nil, "", fmt.Errorf("failed to get results: %w", err)
	}

	task, err := r.db.GetTask(taskID)
	if err != nil {
		return nil, "", fmt.Errorf("failed to get task: %w", err)
	}

	switch format {
	case models.ExportJSON:
		export := map[string]any{
			"task":    task,
			"results": results,
		}
		data, err := json.MarshalIndent(export, "", "  ")
		if err != nil {
			return nil, "", fmt.Errorf("failed to marshal JSON: %w", err)
		}
		return data, "application/json", nil

	case models.ExportCSV:
		var csv strings.Builder
		csv.WriteString("dimension,dataset,metric,value\n")
		for _, result := range results {
			for key, value := range result.Metrics {
				csv.WriteString(fmt.Sprintf("%s,%s,%s,%v\n", result.Dimension, result.DatasetName, key, value))
			}
		}
		return []byte(csv.String()), "text/csv", nil

	case models.ExportPDF:
		return nil, "", fmt.Errorf("PDF export not yet implemented")

	default:
		return nil, "", fmt.Errorf("unsupported export format: %s", format)
	}
}
