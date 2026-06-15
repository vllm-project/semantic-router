package mlpipeline

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

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
	if hasAllAlgorithms(algorithms) {
		runs = []string{"all"}
	}

	return trainExecutionPlan{
		device:       device,
		algorithms:   algorithms,
		runs:         runs,
		cacheDir:     filepath.Join(jobDir, ".cache"),
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

type trainProgressStage struct {
	pct int
	msg string
}

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
