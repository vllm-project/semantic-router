package evaluation

import (
	"bufio"
	"context"
	"encoding/json"
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
func (r *Runner) RunTask(ctx context.Context, taskID string) error {
	task, err := r.db.GetTask(taskID)
	if err != nil {
		return fmt.Errorf("failed to get task: %w", err)
	}
	if task == nil {
		return fmt.Errorf("task not found: %s", taskID)
	}

	// Update status to running
	if err := r.db.UpdateTaskStatus(taskID, models.StatusRunning, ""); err != nil {
		return fmt.Errorf("failed to update task status: %w", err)
	}

	r.sendProgress(taskID, 0, "Starting evaluation", "Initializing evaluation task")

	// Create task-specific output directory
	taskOutputDir := filepath.Join(r.resultsDir, taskID)
	if err := os.MkdirAll(taskOutputDir, 0o755); err != nil {
		_ = r.db.UpdateTaskStatus(taskID, models.StatusFailed, fmt.Sprintf("Failed to create output directory: %v", err))
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	totalDimensions := len(task.Config.Dimensions)
	completedDimensions := 0

	for _, dimension := range task.Config.Dimensions {
		select {
		case <-ctx.Done():
			_ = r.db.UpdateTaskStatus(taskID, models.StatusCancelled, "Task cancelled")
			return ctx.Err()
		default:
		}

		progressBase := (completedDimensions * 100) / totalDimensions
		step := fmt.Sprintf("Evaluating %s", dimension)
		r.sendProgress(taskID, progressBase, step, fmt.Sprintf("Starting %s evaluation", dimension))

		datasets := task.Config.Datasets[string(dimension)]
		if len(datasets) == 0 {
			// Use default dataset
			datasets = []string{"default"}
		}

		for _, dataset := range datasets {
			var result *models.EvaluationResult
			var runErr error

			switch dimension {
			case models.DimensionHallucination:
				result, runErr = r.runHallucinationBenchmark(ctx, taskID, task.Config, dataset, taskOutputDir)
			case models.DimensionReasoning:
				result, runErr = r.runReasoningBenchmark(ctx, taskID, task.Config, dataset, taskOutputDir)
			case models.DimensionAccuracy:
				result, runErr = r.runAccuracyBenchmark(ctx, taskID, task.Config, dataset, taskOutputDir)
			case models.DimensionLatency:
				result, runErr = r.runLatencyBenchmark(ctx, taskID, task.Config)
			case models.DimensionCost:
				result, runErr = r.runCostBenchmark(ctx, taskID, task.Config)
			case models.DimensionSecurity:
				result, runErr = r.runSecurityBenchmark(ctx, taskID, task.Config, dataset, taskOutputDir)
			default:
				log.Printf("Unknown dimension: %s", dimension)
				continue
			}

			if runErr != nil {
				log.Printf("Error running %s benchmark for dataset %s: %v", dimension, dataset, runErr)
				// Continue with other dimensions/datasets
				continue
			}

			if result != nil {
				if err := r.db.SaveResult(result); err != nil {
					log.Printf("Failed to save result: %v", err)
				}

				// Save historical entries for key metrics
				r.saveHistoricalMetrics(result)
			}
		}

		completedDimensions++
		progress := (completedDimensions * 100) / totalDimensions
		r.sendProgress(taskID, progress, step, fmt.Sprintf("Completed %s evaluation", dimension))
	}

	r.sendProgress(taskID, 100, "Completed", "All evaluations finished")
	if err := r.db.UpdateTaskStatus(taskID, models.StatusCompleted, ""); err != nil {
		return fmt.Errorf("failed to update task status: %w", err)
	}

	return nil
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

// ensureV1Suffix ensures the endpoint has /v1 suffix for OpenAI client compatibility.
func ensureV1Suffix(endpoint string) string {
	endpoint = strings.TrimSuffix(endpoint, "/")
	if !strings.HasSuffix(endpoint, "/v1") {
		endpoint += "/v1"
	}
	return endpoint
}

// runHallucinationBenchmark runs the hallucination detection benchmark.
func (r *Runner) runHallucinationBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig, dataset, outputDir string) (*models.EvaluationResult, error) {
	outputPath := filepath.Join(outputDir, fmt.Sprintf("hallucination_%s.json", dataset))

	// Ensure endpoint has /v1 suffix for OpenAI client compatibility
	endpoint := ensureV1Suffix(cfg.Endpoint)

	// Build command arguments
	args := []string{
		"-m", "bench.hallucination.evaluate",
		"--endpoint", endpoint,
		"--dataset", dataset,
		"--max-samples", fmt.Sprintf("%d", cfg.MaxSamples),
		"--output-dir", outputDir,
		"--quiet",
	}

	if cfg.Model != "" {
		args = append(args, "--model", cfg.Model)
	}

	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath is configured at startup, not user input
	cmd.Dir = r.projectRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH="+r.projectRoot)

	r.activeProcesses.Store(taskID, cmd)
	defer r.activeProcesses.Delete(taskID)

	output, err := r.runCommandWithProgress(ctx, cmd, taskID, "hallucination")
	if err != nil {
		return nil, fmt.Errorf("hallucination benchmark failed: %w", err)
	}

	// Parse output JSON
	metrics, err := ParseHallucinationOutput(output, outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse hallucination output: %w", err)
	}

	return &models.EvaluationResult{
		TaskID:         taskID,
		Dimension:      models.DimensionHallucination,
		DatasetName:    dataset,
		Metrics:        metrics,
		RawResultsPath: outputPath,
	}, nil
}

// runReasoningBenchmark runs the reasoning mode evaluation benchmark.
func (r *Runner) runReasoningBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig, dataset, outputDir string) (*models.EvaluationResult, error) {
	// Build command arguments
	// Note: --generate-plots and --generate-report use action="store_true" with default=True
	// so we don't need to pass them explicitly
	// Ensure endpoint has /v1 suffix for OpenAI client compatibility
	endpoint := ensureV1Suffix(cfg.Endpoint)
	args := []string{
		"-m", "bench.reasoning.reasoning_mode_eval",
		"--endpoint", endpoint,
		"--datasets", dataset,
		"--samples-per-category", fmt.Sprintf("%d", cfg.SamplesPerCat),
		"--output-dir", outputDir,
	}

	if cfg.Model != "" {
		args = append(args, "--model", cfg.Model)
	}

	if cfg.Concurrent > 0 {
		args = append(args, "--concurrent-requests", fmt.Sprintf("%d", cfg.Concurrent))
	}

	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath is configured at startup, not user input
	cmd.Dir = r.projectRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH="+r.projectRoot)

	r.activeProcesses.Store(taskID, cmd)
	defer r.activeProcesses.Delete(taskID)

	output, err := r.runCommandWithProgress(ctx, cmd, taskID, "reasoning")
	if err != nil {
		return nil, fmt.Errorf("reasoning benchmark failed: %w", err)
	}

	// Parse output JSON
	summaryPath := filepath.Join(outputDir, "reasoning_mode_eval_summary.json")
	metrics, err := ParseReasoningOutput(output, summaryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse reasoning output: %w", err)
	}

	return &models.EvaluationResult{
		TaskID:         taskID,
		Dimension:      models.DimensionReasoning,
		DatasetName:    dataset,
		Metrics:        metrics,
		RawResultsPath: summaryPath,
	}, nil
}

// runAccuracyBenchmark runs the accuracy evaluation using the router benchmark.
func (r *Runner) runAccuracyBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig, dataset, outputDir string) (*models.EvaluationResult, error) {
	outputPath := filepath.Join(outputDir, fmt.Sprintf("accuracy_%s.json", dataset))

	// Ensure endpoint has /v1 suffix for OpenAI client compatibility
	endpoint := ensureV1Suffix(cfg.Endpoint)
	if endpoint == "/v1" {
		endpoint = "http://localhost:8801/v1"
	}

	// Model to test - default to MoM
	model := cfg.Model
	if model == "" {
		model = "MoM"
	}

	args := []string{
		"-m", "bench.reasoning.router_reason_bench_multi_dataset",
		"--dataset", dataset, // Note: singular --dataset, not --datasets
		"--samples-per-category", fmt.Sprintf("%d", cfg.SamplesPerCat),
		"--output-dir", outputDir,
		"--run-router", // Flag to run against router
		"--router-endpoint", endpoint,
		"--router-models", model,
	}

	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath is configured at startup, not user input
	cmd.Dir = r.projectRoot
	cmd.Env = append(os.Environ(), "PYTHONPATH="+r.projectRoot)

	r.activeProcesses.Store(taskID, cmd)
	defer r.activeProcesses.Delete(taskID)

	output, err := r.runCommandWithProgress(ctx, cmd, taskID, "accuracy")
	if err != nil {
		return nil, fmt.Errorf("accuracy benchmark failed: %w", err)
	}

	metrics, err := ParseAccuracyOutput(output, outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse accuracy output: %w", err)
	}

	return &models.EvaluationResult{
		TaskID:         taskID,
		Dimension:      models.DimensionAccuracy,
		DatasetName:    dataset,
		Metrics:        metrics,
		RawResultsPath: outputPath,
	}, nil
}

// runLatencyBenchmark queries Prometheus for latency metrics.
func (r *Runner) runLatencyBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig) (*models.EvaluationResult, error) {
	// Query Prometheus for latency metrics
	// This is a placeholder - in a real implementation, you'd query the embedded Prometheus
	metrics := map[string]interface{}{
		"avg_latency_ms": 0.0,
		"p50_latency_ms": 0.0,
		"p95_latency_ms": 0.0,
		"p99_latency_ms": 0.0,
		"ttft_avg_ms":    0.0,
		"tpot_avg_ms":    0.0,
		"source":         "prometheus",
		"timestamp":      time.Now().Format(time.RFC3339),
		"status":         "not_implemented",
		"message":        "Latency benchmarks require Prometheus integration",
	}

	r.sendProgress(taskID, 50, "latency", "Latency metrics collection not yet implemented")

	return &models.EvaluationResult{
		TaskID:      taskID,
		Dimension:   models.DimensionLatency,
		DatasetName: "prometheus",
		Metrics:     metrics,
	}, nil
}

// runCostBenchmark queries Prometheus for cost metrics.
func (r *Runner) runCostBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig) (*models.EvaluationResult, error) {
	// Query Prometheus for cost metrics
	metrics := map[string]interface{}{
		"total_cost":       0.0,
		"cost_per_request": 0.0,
		"cost_by_model":    map[string]float64{},
		"source":           "prometheus",
		"timestamp":        time.Now().Format(time.RFC3339),
		"status":           "not_implemented",
		"message":          "Cost benchmarks require Prometheus integration",
	}

	r.sendProgress(taskID, 50, "cost", "Cost metrics collection not yet implemented")

	return &models.EvaluationResult{
		TaskID:      taskID,
		Dimension:   models.DimensionCost,
		DatasetName: "prometheus",
		Metrics:     metrics,
	}, nil
}

// runSecurityBenchmark runs the jailbreak detection benchmark.
func (r *Runner) runSecurityBenchmark(ctx context.Context, taskID string, cfg models.EvaluationConfig, dataset, outputDir string) (*models.EvaluationResult, error) {
	// Security benchmarks use the jailbreak classifier
	metrics := map[string]interface{}{
		"detection_rate":      0.0,
		"false_positive_rate": 0.0,
		"source":              "jailbreak_classifier",
		"timestamp":           time.Now().Format(time.RFC3339),
		"status":              "not_implemented",
		"message":             "Security benchmarks require jailbreak classifier integration",
	}

	r.sendProgress(taskID, 50, "security", "Security metrics collection not yet implemented")

	return &models.EvaluationResult{
		TaskID:      taskID,
		Dimension:   models.DimensionSecurity,
		DatasetName: dataset,
		Metrics:     metrics,
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

	// Helper to parse tqdm progress from a line
	parseProgress := func(line string) {
		// tqdm format: " 50%|████████  | 25/50 [00:30<00:30, 1.00it/s]"
		// Also handle: "50%|..."
		if strings.Contains(line, "%|") || strings.Contains(line, "% |") {
			// Find percentage - look for number followed by %
			for i := 0; i < len(line); i++ {
				if line[i] == '%' && i > 0 {
					// Find the start of the number
					start := i - 1
					for start > 0 && (line[start-1] >= '0' && line[start-1] <= '9') {
						start--
					}
					if start < i {
						var percent int
						_, _ = fmt.Sscanf(line[start:i], "%d", &percent)
						if percent > 0 && percent <= 100 {
							r.sendProgress(taskID, percent, dimension, fmt.Sprintf("Processing: %d%%", percent))
						}
					}
					break
				}
			}
		}
	}

	// Read stdout
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			output.WriteString(line + "\n")
			parseProgress(line)
		}
	}()

	// Read stderr - tqdm writes progress here
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			errOutput.WriteString(line + "\n")
			parseProgress(line)
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
		string(models.DimensionHallucination): {
			{Name: "halueval", Description: "HaluEval hallucination detection benchmark", Dimension: models.DimensionHallucination},
		},
		string(models.DimensionReasoning): {
			{Name: "mmlu", Description: "Massive Multitask Language Understanding", Dimension: models.DimensionReasoning},
			{Name: "gpqa", Description: "Graduate-level Google-Proof Q&A", Dimension: models.DimensionReasoning},
		},
		string(models.DimensionAccuracy): {
			{Name: "mmlu", Description: "MMLU benchmark for accuracy testing", Dimension: models.DimensionAccuracy},
			{Name: "gpqa", Description: "GPQA benchmark for accuracy testing", Dimension: models.DimensionAccuracy},
		},
		string(models.DimensionLatency): {
			{Name: "prometheus", Description: "Prometheus metrics (live data)", Dimension: models.DimensionLatency},
		},
		string(models.DimensionCost): {
			{Name: "prometheus", Description: "Prometheus cost metrics (live data)", Dimension: models.DimensionCost},
		},
		string(models.DimensionSecurity): {
			{Name: "jailbreak", Description: "Jailbreak detection test set", Dimension: models.DimensionSecurity},
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
		export := map[string]interface{}{
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
