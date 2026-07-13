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
)

// runBenchmarkSubprocess runs Layer 1: benchmark.py as a local subprocess.
func (r *Runner) runBenchmarkSubprocess(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest, release func()) (string, error) {
	job, err := r.createJob("benchmark")
	if err != nil {
		release()
		return "", err
	}
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		r.failJob(job.ID, "benchmark output directory could not be created")
		release()
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	if err := r.setJobRunning(job); err != nil {
		release()
		return "", err
	}

	r.startAsyncJob(job.ID, release, func() {
		r.executeBenchmarkSubprocess(ctx, job.ID, jobDir, modelsYAMLPath, queryJSONLPath, req)
	})
	return job.ID, nil
}

func (r *Runner) executeBenchmarkSubprocess(ctx context.Context, jobID, jobDir, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) {
	defer r.cleanupManagedUploads(modelsYAMLPath, queryJSONLPath)
	jobCtx, cancel := context.WithTimeout(ctx, mlBenchmarkJobTimeout)
	defer cancel()
	r.sendProgress(jobID, 5, "Starting benchmark", "Preparing benchmark run")
	outputFile := filepath.Join(jobDir, "benchmark_output.jsonl")
	args, concurrency := r.buildBenchmarkArgs(req, modelsYAMLPath, queryJSONLPath, outputFile)
	r.sendProgress(jobID, 10, "Running benchmark", fmt.Sprintf("Running benchmark.py with concurrency=%d", concurrency))

	numQueries := countFileLines(queryJSONLPath)
	numModels := countYAMLModels(modelsYAMLPath)
	log.Printf("[benchmark/%s] Expecting ~%d results (%d queries × %d models)", jobID, numQueries*numModels, numQueries, numModels)

	err := r.runBenchmarkCommand(jobCtx, jobID, args)
	if err != nil {
		r.failJob(jobID, "benchmark execution failed")
		r.sendProgress(jobID, 100, "Failed", "Benchmark execution failed")
		return
	}

	if _, err := os.Stat(outputFile); err != nil {
		r.failJob(jobID, "benchmark output file not created")
		r.sendProgress(jobID, 100, "Failed", "Output file not found")
		return
	}

	if err := r.completeJob(jobID, []string{outputFile}); err != nil {
		r.failJob(jobID, "benchmark output was rejected")
		r.sendProgress(jobID, 100, "Failed", "Benchmark output was rejected")
		return
	}
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

func (r *Runner) runBenchmarkCommand(ctx context.Context, jobID string, args []string) error {
	cmd := exec.CommandContext(ctx, r.pythonPath, args...) //nolint:gosec // pythonPath and args are server-controlled, not user input
	cmd.Dir = r.trainingDir
	cmd.Env = append(os.Environ(),
		"PYTHONIOENCODING=utf-8",
		"PYTHONUNBUFFERED=1",
		"HF_HUB_DISABLE_PROGRESS_BARS=1",
	)

	pipeR, pipeW, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("failed to create pipe: %w", err)
	}
	cmd.Stdout = pipeW
	cmd.Stderr = pipeW
	if err := cmd.Start(); err != nil {
		pipeW.Close()
		pipeR.Close()
		return fmt.Errorf("failed to start benchmark: %w", err)
	}
	pipeW.Close()

	scanErr := r.collectBenchmarkProgress(jobID, pipeR)
	waitErr := cmd.Wait()
	if scanErr != nil {
		return scanErr
	}
	if waitErr != nil {
		return waitErr
	}
	return nil
}

func (r *Runner) collectBenchmarkProgress(jobID string, pipeR *os.File) error {
	defer pipeR.Close()
	tqdmProgressRe := regexp.MustCompile(`\b(\d+)/(\d+)\b`)
	scanner := bufio.NewScanner(pipeR)
	scanner.Split(scanCRLF)
	scanner.Buffer(make([]byte, 4096), 64<<10)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		r.updateBenchmarkProgressFromLine(jobID, line, tqdmProgressRe)
	}
	if scanner.Err() != nil {
		return fmt.Errorf("benchmark emitted an invalid progress stream")
	}
	return nil
}

func (r *Runner) updateBenchmarkProgressFromLine(jobID, line string, progressRe *regexp.Regexp) {
	matches := progressRe.FindStringSubmatch(line)
	if len(matches) != 3 {
		return
	}
	current, err1 := strconv.Atoi(matches[1])
	total, err2 := strconv.Atoi(matches[2])
	if err1 != nil || err2 != nil || total <= 0 {
		return
	}
	tqdmPct := current * 100 / total
	pct := 10 + tqdmPct*85/100
	if pct > 95 {
		pct = 95
	}
	msg := fmt.Sprintf("Benchmarking: %d/%d queries completed (%d%%)", current, total, tqdmPct)
	r.sendProgress(jobID, pct, "Running benchmark", msg)
}
