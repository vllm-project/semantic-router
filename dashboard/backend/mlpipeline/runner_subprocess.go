package mlpipeline

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// runBenchmarkSubprocess runs Layer 1: benchmark.py as a local subprocess.
func (r *Runner) runBenchmarkSubprocess(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) (string, error) {
	job := r.createJob("benchmark")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.setJobRunning(job)

	go r.executeBenchmarkSubprocess(ctx, job.ID, jobDir, modelsYAMLPath, queryJSONLPath, req)
	return job.ID, nil
}

func (r *Runner) executeBenchmarkSubprocess(ctx context.Context, jobID, jobDir, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) {
	r.sendProgress(jobID, 5, "Starting benchmark", "Preparing benchmark run")
	outputFile := filepath.Join(jobDir, "benchmark_output.jsonl")
	args, concurrency := r.buildBenchmarkArgs(req, modelsYAMLPath, queryJSONLPath, outputFile)
	r.sendProgress(jobID, 10, "Running benchmark", fmt.Sprintf("Running benchmark.py with concurrency=%d", concurrency))

	numQueries := countFileLines(queryJSONLPath)
	numModels := countYAMLModels(modelsYAMLPath)
	log.Printf("[benchmark/%s] Expecting ~%d results (%d queries × %d models)", jobID, numQueries*numModels, numQueries, numModels)

	fullOutput, err := r.runBenchmarkCommand(ctx, jobID, args)
	if err != nil {
		r.failJob(jobID, fmt.Sprintf("benchmark failed: %v", err))
		r.sendProgress(jobID, 100, "Failed", err.Error())
		return
	}
	log.Printf("[benchmark/%s] Output:\n%s", jobID, fullOutput)

	// Verify output exists
	if _, err := os.Stat(outputFile); err != nil {
		r.failJob(jobID, "benchmark output file not created")
		r.sendProgress(jobID, 100, "Failed", "Output file not found")
		return
	}

	r.completeJob(jobID, []string{outputFile})
	r.sendProgress(jobID, 100, "Completed", "Benchmark finished successfully")
}

func (r *Runner) buildBenchmarkArgs(req BenchmarkRequest, modelsYAMLPath, queryJSONLPath, outputFile string) ([]string, int) {
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
	return args, concurrency
}

func (r *Runner) runBenchmarkCommand(ctx context.Context, jobID string, args []string) (string, error) {
	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath and args are server-controlled, not user input
	cmd.Dir = r.trainingDir
	cmd.Env = append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"PYTHONUNBUFFERED=1",
		"HF_HUB_DISABLE_PROGRESS_BARS=1",
	)

	pipeR, pipeW, err := os.Pipe()
	if err != nil {
		return "", fmt.Errorf("failed to create pipe: %w", err)
	}
	cmd.Stdout = pipeW
	cmd.Stderr = pipeW
	if err := cmd.Start(); err != nil {
		pipeW.Close()
		pipeR.Close()
		return "", fmt.Errorf("failed to start benchmark: %w", err)
	}
	pipeW.Close()

	fullOutput := r.collectBenchmarkProgress(jobID, pipeR)
	waitErr := cmd.Wait()
	if waitErr != nil {
		return fullOutput, waitErr
	}
	return fullOutput, nil
}

func (r *Runner) collectBenchmarkProgress(jobID string, pipeR *os.File) string {
	defer pipeR.Close()
	tqdmProgressRe := regexp.MustCompile(`\b(\d+)/(\d+)\b`)
	scanner := bufio.NewScanner(pipeR)
	scanner.Split(scanCRLF)

	var outputBuf strings.Builder
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		outputBuf.WriteString(line + "\n")
		r.updateBenchmarkProgressFromLine(jobID, line, tqdmProgressRe)
	}
	return outputBuf.String()
}

func (r *Runner) updateBenchmarkProgressFromLine(jobID, line string, progressRe *regexp.Regexp) {
	// Parse tqdm progress: look for "X/Y" pattern (e.g. "5/20")
	matches := progressRe.FindStringSubmatch(line)
	if len(matches) != 3 {
		return
	}
	current, err1 := strconv.Atoi(matches[1])
	total, err2 := strconv.Atoi(matches[2])
	if err1 != nil || err2 != nil || total <= 0 {
		return
	}
	// Map tqdm 0-100% to our 10-95% range
	tqdmPct := current * 100 / total
	pct := 10 + tqdmPct*85/100
	if pct > 95 {
		pct = 95
	}
	msg := fmt.Sprintf("Benchmarking: %d/%d queries completed (%d%%)", current, total, tqdmPct)
	r.sendProgress(jobID, pct, "Running benchmark", msg)
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
	jobDir := r.TrainDir()
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.setJobRunning(job)

	go r.executeTrainSubprocess(ctx, job.ID, jobDir, benchmarkDataPath, req)
	return job.ID, nil
}

func (r *Runner) executeTrainSubprocess(ctx context.Context, jobID, jobDir, benchmarkDataPath string, req TrainRequest) {
	r.sendProgress(jobID, 5, "Starting training", "Preparing training run")
	plan := r.buildTrainExecutionPlan(jobDir, req)
	r.sendProgress(jobID, 10, "Running training", plan.startMessage)

	done := r.startTrainProgressTicker(jobID, r.buildTrainProgressStages(plan.algorithms))
	defer close(done)

	lastErr := r.runTrainAlgorithms(ctx, jobID, benchmarkDataPath, plan)
	if lastErr != nil && len(plan.runs) == 1 {
		r.failJob(jobID, lastErr.Error())
		r.sendProgress(jobID, 100, "Failed", lastErr.Error())
		return
	}

	outputFiles := collectTrainOutputFiles(jobDir)
	if len(outputFiles) == 0 {
		errMsg := "no model files were generated"
		if lastErr != nil {
			errMsg = lastErr.Error()
		}
		r.failJob(jobID, errMsg)
		r.sendProgress(jobID, 100, "Failed", errMsg)
		return
	}

	r.completeJob(jobID, outputFiles)
	r.sendProgress(jobID, 100, "Completed", fmt.Sprintf("Training finished: %d model(s) generated", len(outputFiles)))
}

type trainExecutionPlan struct {
	device       string
	algorithms   []string
	runs         []string
	cacheDir     string
	pythonEnv    []string
	commonArgs   []string
	startMessage string
}

func (r *Runner) buildTrainExecutionPlan(jobDir string, req TrainRequest) trainExecutionPlan {
	device := req.Device
	if device == "" {
		device = "cpu"
	}

	algorithms := req.Algorithms
	if len(algorithms) == 0 {
		algorithms = []string{"knn", "kmeans", "svm", "mlp"}
	}

	runs := algorithms

	// Check if all 4 algorithms are selected → use single --algorithm all invocation
	if hasAllAlgorithms(algorithms) {
		runs = []string{"all"}
	}

	return trainExecutionPlan{
		device:     device,
		algorithms: algorithms,
		runs:       runs,
		// Shared cache dir so embeddings are computed once and reused across runs
		cacheDir: filepath.Join(jobDir, ".cache"),
		// Environment for all Python invocations
		pythonEnv:    trainPythonEnv(),
		commonArgs:   buildTrainCommonArgs(req),
		startMessage: fmt.Sprintf("Training %s on device=%s", strings.Join(algorithms, ", "), device),
	}
}

func hasAllAlgorithms(algorithms []string) bool {
	if len(algorithms) != 4 {
		return false
	}
	allSet := map[string]bool{}
	for _, a := range algorithms {
		allSet[a] = true
	}
	return allSet["knn"] && allSet["kmeans"] && allSet["svm"] && allSet["mlp"]
}

func trainPythonEnv() []string {
	return append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"PYTHONUNBUFFERED=1",
		"TQDM_DISABLE=1",
		"HF_HUB_DISABLE_PROGRESS_BARS=1",
	)
}

func buildTrainCommonArgs(req TrainRequest) []string {
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
	return commonArgs
}

func (r *Runner) runTrainAlgorithms(ctx context.Context, jobID, benchmarkDataPath string, plan trainExecutionPlan) error {
	// Execute train.py for each run (single "all" run, or one per algorithm)
	var lastErr error
	for _, algFlag := range plan.runs {
		args := []string{
			filepath.Join(r.trainingDir, "train.py"),
			"--data-file", benchmarkDataPath,
			"--output-dir", r.TrainDir(),
			"--device", plan.device,
			"--algorithm", algFlag,
			"--cache-dir", plan.cacheDir,
		}
		args = append(args, plan.commonArgs...)
		log.Printf("[train/%s] Running: %s %v", jobID, r.pythonPath, args)

		cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath and args are server-controlled, not user input
		cmd.Dir = r.trainingDir
		cmd.Env = plan.pythonEnv

		output, err := cmd.CombinedOutput()
		log.Printf("[train/%s] Output (algorithm=%s):\n%s", jobID, algFlag, string(output))
		if err != nil {
			lastErr = fmt.Errorf("training %s failed: %w\n%s", algFlag, err, string(output))
			log.Printf("[train/%s] Error training %s: %v", jobID, algFlag, err)
			// Continue with remaining algorithms instead of aborting
		}
	}
	return lastErr
}

func collectTrainOutputFiles(jobDir string) []string {
	outputFiles := []string{}
	for _, modelName := range []string{"knn_model.json", "kmeans_model.json", "svm_model.json", "mlp_model.json"} {
		modelFile := filepath.Join(jobDir, modelName)
		if _, err := os.Stat(modelFile); err == nil {
			outputFiles = append(outputFiles, modelFile)
		}
	}
	return outputFiles
}

func (r *Runner) startTrainProgressTicker(jobID string, stages []trainProgressStage) chan struct{} {
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
					r.sendProgress(jobID, stages[idx].pct, "Training", stages[idx].msg)
					idx++
				}
			}
		}
	}()
	return done
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

	if len(algStages) > 0 {
		step := 35 / len(algStages)
		for i := range algStages {
			algStages[i].pct = 55 + i*step
		}
		stages = append(stages, algStages...)
	}

	stages = append(stages, trainProgressStage{90, "Finalizing models..."})
	return stages
}
