package mlpipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

type trainRequestPaths struct {
	OutputDir string `json:"output_dir"`
	CacheDir  string `json:"cache_dir"`
}

func TestRunTrainHTTPConcurrentJobsUseDistinctDirectories(t *testing.T) {
	requests := make(chan trainRequestPaths, 2)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var paths trainRequestPaths
		if err := json.NewDecoder(request.Body).Decode(&paths); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		requests <- paths
		<-release

		if err := os.MkdirAll(paths.OutputDir, 0o755); err != nil {
			http.Error(writer, err.Error(), http.StatusInternalServerError)
			return
		}
		outputFile := filepath.Join(paths.OutputDir, "knn_model.json")
		if err := os.WriteFile(outputFile, []byte("{}"), 0o600); err != nil {
			http.Error(writer, err.Error(), http.StatusInternalServerError)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(writer, "data: {\"percent\":100,\"step\":\"completed\",\"done\":true,\"success\":true,\"output_files\":[%q]}\n\n", outputFile)
	}))
	defer server.Close()

	runner := newTrainIsolationTestRunner(t, RunnerConfig{MLServiceURL: server.URL})
	firstID, err := runner.runTrainHTTP(context.Background(), "first.jsonl", TrainRequest{Algorithms: []string{"knn"}})
	if err != nil {
		t.Fatal(err)
	}
	secondID, err := runner.runTrainHTTP(context.Background(), "second.jsonl", TrainRequest{Algorithms: []string{"knn"}})
	if err != nil {
		t.Fatal(err)
	}

	firstRequest := <-requests
	secondRequest := <-requests
	close(release)
	if firstRequest.OutputDir == secondRequest.OutputDir {
		t.Fatalf("concurrent jobs share output directory %q", firstRequest.OutputDir)
	}
	if firstRequest.CacheDir == secondRequest.CacheDir {
		t.Fatalf("concurrent jobs share cache directory %q", firstRequest.CacheDir)
	}

	requestsByOutputDir := map[string]trainRequestPaths{
		firstRequest.OutputDir:  firstRequest,
		secondRequest.OutputDir: secondRequest,
	}
	for _, jobID := range []string{firstID, secondID} {
		outputDir := runner.JobDir(jobID)
		requestPaths, ok := requestsByOutputDir[outputDir]
		if !ok {
			t.Fatalf("job %q did not use output directory %q", jobID, outputDir)
		}
		if requestPaths.CacheDir != filepath.Join(outputDir, ".cache") {
			t.Fatalf("job %q cache directory = %q, want %q", jobID, requestPaths.CacheDir, filepath.Join(outputDir, ".cache"))
		}
		assertCompletedTrainJob(t, runner, jobID, outputDir)
	}
}

func TestRunTrainSubprocessConcurrentJobsUseDistinctDirectories(t *testing.T) {
	root := t.TempDir()
	trainingDir := filepath.Join(root, "training")
	if err := os.MkdirAll(trainingDir, 0o755); err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(root, "fake-python")
	script := `#!/bin/sh
output_dir=
cache_dir=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    --cache-dir) cache_dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$output_dir" "$cache_dir"
printf '{}' > "$output_dir/knn_model.json"
printf 'cached' > "$cache_dir/marker"
`
	if err := os.WriteFile(scriptPath, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}

	runner := newTrainIsolationTestRunner(t, RunnerConfig{
		TrainingDir: trainingDir,
		PythonPath:  scriptPath,
	})
	firstID, err := runner.runTrainSubprocess(context.Background(), "first.jsonl", TrainRequest{Algorithms: []string{"knn"}})
	if err != nil {
		t.Fatal(err)
	}
	secondID, err := runner.runTrainSubprocess(context.Background(), "second.jsonl", TrainRequest{Algorithms: []string{"knn"}})
	if err != nil {
		t.Fatal(err)
	}

	firstJob := waitForTrainJob(t, runner, firstID)
	secondJob := waitForTrainJob(t, runner, secondID)
	firstDir := filepath.Dir(firstJob.OutputFiles[0])
	secondDir := filepath.Dir(secondJob.OutputFiles[0])
	if firstDir == secondDir {
		t.Fatalf("concurrent jobs share output directory %q", firstDir)
	}
	for _, dir := range []string{firstDir, secondDir} {
		if _, err := os.Stat(filepath.Join(dir, ".cache", "marker")); err != nil {
			t.Fatalf("cache marker for %q: %v", dir, err)
		}
	}
}

func newTrainIsolationTestRunner(t *testing.T, cfg RunnerConfig) *Runner {
	t.Helper()
	root := t.TempDir()
	store, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		store.Close()
	})
	cfg.DataDir = filepath.Join(root, "data")
	cfg.Workflow = store
	runner, err := NewRunner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return runner
}

func assertCompletedTrainJob(t *testing.T, runner *Runner, jobID, outputDir string) {
	t.Helper()
	job := waitForTrainJob(t, runner, jobID)
	if len(job.OutputFiles) != 1 || filepath.Dir(job.OutputFiles[0]) != outputDir {
		t.Fatalf("job %q outputs = %v, want one file under %q", jobID, job.OutputFiles, outputDir)
	}
}

func waitForTrainJob(t *testing.T, runner *Runner, jobID string) *Job {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		job := runner.GetJob(jobID)
		if job != nil && job.Status == StatusCompleted {
			return job
		}
		if job != nil && job.Status == StatusFailed {
			t.Fatalf("job %q failed: %s", jobID, job.Error)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job %q did not complete", jobID)
	return nil
}
