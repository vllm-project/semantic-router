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

		// Get datasets for this dimension
		datasets := task.Config.Datasets[string(dimension)]
		if len(datasets) == 0 {
			// Use default dataset based on dimension
			datasets = []string{getDefaultDataset(dimension)}
		}

		// Evaluate each dataset for this dimension
		for _, dataset := range datasets {
			var result *models.EvaluationResult
			var runErr error

			switch dimension {
			case models.DimensionDomain, models.DimensionFactCheck, models.DimensionUserFeedback:
				result, runErr = r.runSignalEvaluation(ctx, taskID, task.Config, string(dimension), dataset, taskOutputDir)
			default:
				log.Printf("Unknown dimension: %s", dimension)
				continue
			}

			if runErr != nil {
				log.Printf("Error running %s evaluation on dataset %s: %v", dimension, dataset, runErr)
				// Continue with other datasets
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

// getDefaultDataset returns the default dataset ID for a given dimension.
func getDefaultDataset(dimension models.EvaluationDimension) string {
	switch dimension {
	case models.DimensionDomain:
		return "mmlu-pro-en"
	case models.DimensionFactCheck:
		return "fact-check-en"
	case models.DimensionUserFeedback:
		return "feedback-en"
	default:
		return "default"
	}
}

// runSignalEvaluation runs the signal evaluation for a specific dataset.
func (r *Runner) runSignalEvaluation(ctx context.Context, taskID string, cfg models.EvaluationConfig, dimension, datasetID, outputDir string) (*models.EvaluationResult, error) {
	outputPath := filepath.Join(outputDir, fmt.Sprintf("signal_eval_%s.json", datasetID))

	// Use endpoint as-is for eval API
	endpoint := strings.TrimSuffix(cfg.Endpoint, "/")

	// Build command arguments
	args := []string{
		"src/training/model_eval/signal_eval.py",
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
	cmd.Env = append(os.Environ(), "PYTHONPATH="+r.projectRoot)

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
