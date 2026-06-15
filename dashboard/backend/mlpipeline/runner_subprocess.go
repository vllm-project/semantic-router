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
