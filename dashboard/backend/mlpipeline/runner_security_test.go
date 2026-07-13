package mlpipeline

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func newPipelineSecurityTestRunner(t *testing.T, serviceURL string) (*Runner, string) {
	t.Helper()
	root := t.TempDir()
	wf, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatalf("open workflow store: %v", err)
	}
	t.Cleanup(func() { _ = wf.Close() })
	dataDir := filepath.Join(root, "pipeline-data")
	runner, err := NewRunner(RunnerConfig{
		DataDir:      dataDir,
		TrainingDir:  t.TempDir(),
		PythonPath:   "python3",
		MLServiceURL: serviceURL,
		Workflow:     wf,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	return runner, runner.dataDir
}

func writePipelineTestInputs(t *testing.T, runner *Runner) (string, string, string) {
	t.Helper()
	uploadDir, err := runner.CreateUploadDir("inputs-")
	if err != nil {
		t.Fatalf("CreateUploadDir: %v", err)
	}
	modelsPath := filepath.Join(uploadDir, "models.yaml")
	queriesPath := filepath.Join(uploadDir, "queries.jsonl")
	if err := os.WriteFile(modelsPath, []byte("models:\n  - name: test-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(queriesPath, []byte("{\"query\":\"hello\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return uploadDir, modelsPath, queriesPath
}

func waitForPipelineJob(t *testing.T, runner *Runner, jobID string, status JobStatus) *Job {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		job := runner.GetJob(jobID)
		if job != nil && job.Status == status {
			return job
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job %s did not reach status %s; latest=%+v", jobID, status, runner.GetJob(jobID))
	return nil
}

func waitForPipelineRunnerIdle(t *testing.T, runner *Runner) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runner.admissionMu.Lock()
		activeJobs := runner.activeJobs
		runner.admissionMu.Unlock()
		if activeJobs == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("ML pipeline runner did not release its active job")
}

func TestManagedPipelineFilesStayInsidePrivateDataRoot(t *testing.T) {
	runner, dataDir := newPipelineSecurityTestRunner(t, "")
	uploadDir, err := runner.CreateUploadDir("secure-")
	if err != nil {
		t.Fatal(err)
	}
	uploadInfo, err := os.Stat(uploadDir)
	if err != nil {
		t.Fatal(err)
	}
	if got := uploadInfo.Mode().Perm(); got != 0o700 {
		t.Fatalf("upload dir mode = %o, want 700", got)
	}

	managedFile := filepath.Join(uploadDir, "input.jsonl")
	if writeErr := os.WriteFile(managedFile, []byte("{}\n"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	validated, err := runner.ValidateManagedFile(managedFile)
	if err != nil {
		t.Fatalf("ValidateManagedFile: %v", err)
	}
	if !pathWithinRoot(dataDir, validated) {
		t.Fatalf("validated path %q escaped %q", validated, dataDir)
	}

	outside := filepath.Join(t.TempDir(), "outside.jsonl")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.ValidateManagedFile(outside); err == nil {
		t.Fatal("outside file unexpectedly accepted")
	}
	symlink := filepath.Join(uploadDir, "symlink.jsonl")
	if err := os.Symlink(outside, symlink); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.ValidateManagedFile(symlink); err == nil {
		t.Fatal("symlink unexpectedly accepted")
	}
	if err := runner.RemoveUploadDir(filepath.Dir(dataDir)); err == nil {
		t.Fatal("outside directory removal unexpectedly accepted")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("outside file was modified: %v", err)
	}
}

func TestPipelineRequestValidationBoundsWork(t *testing.T) {
	benchmarkCases := []BenchmarkRequest{
		{Concurrency: -1},
		{Concurrency: 17},
		{MaxTokens: 8193},
		{Temperature: 2.1},
		{Limit: 20_001},
	}
	for _, req := range benchmarkCases {
		if err := ValidateBenchmarkRequest(req); err == nil {
			t.Fatalf("invalid benchmark request accepted: %+v", req)
		}
	}
	if err := ValidateBenchmarkRequest(BenchmarkRequest{Concurrency: 16, MaxTokens: 8192, Temperature: 2, Limit: 20_000}); err != nil {
		t.Fatalf("valid benchmark request rejected: %v", err)
	}

	trainCases := []TrainRequest{
		{Algorithms: []string{"shell"}},
		{Algorithms: []string{"knn", "knn"}},
		{Device: "remote"},
		{EmbeddingModel: "arbitrary/model"},
		{BatchSize: 1025},
		{SvmKernel: "poly"},
		{MlpHiddenSizes: "256,--inject"},
		{MlpDropout: 1.1},
	}
	for _, req := range trainCases {
		if err := ValidateTrainRequest(req); err == nil {
			t.Fatalf("invalid train request accepted: %+v", req)
		}
	}
	validTrain := TrainRequest{
		Algorithms: []string{"knn", "kmeans", "svm", "mlp"}, Device: "cpu", EmbeddingModel: "qwen3",
		QualityWeight: 0.9, BatchSize: 32, KnnK: 5, KmeansClusters: 8,
		SvmKernel: "rbf", SvmGamma: 1, MlpHiddenSizes: "256,128", MlpEpochs: 100,
		MlpLearningRate: 0.001, MlpDropout: 0.1,
	}
	if err := ValidateTrainRequest(validTrain); err != nil {
		t.Fatalf("valid train request rejected: %v", err)
	}

	invalidConfig := ConfigRequest{Decisions: []DecisionEntry{{Name: "bad\nname", Algorithm: "knn"}}}
	if err := ValidateConfigRequest(invalidConfig); err == nil {
		t.Fatal("config with control character unexpectedly accepted")
	}
	for _, invalidConfig := range []ConfigRequest{
		{ModelsPath: "   "},
		{ModelsPath: " /data/models"},
		{Decisions: []DecisionEntry{{Name: " trailing ", Algorithm: "knn"}}},
	} {
		if err := ValidateConfigRequest(invalidConfig); err == nil {
			t.Fatalf("non-canonical config string unexpectedly accepted: %+v", invalidConfig)
		}
	}
}

func TestBenchmarkWorkloadUsesFileAndCombinedBudgets(t *testing.T) {
	runner, _ := newPipelineSecurityTestRunner(t, "")
	_, modelsPath, queriesPath := writePipelineTestInputs(t, runner)
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{}); err != nil {
		t.Fatalf("small workload rejected: %v", err)
	}

	var tooManyQueries strings.Builder
	for range maxBenchmarkQueries + 1 {
		tooManyQueries.WriteString("{}\n")
	}
	if err := os.WriteFile(queriesPath, []byte(tooManyQueries.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{}); err == nil {
		t.Fatal("query record amplification unexpectedly accepted")
	}

	if err := os.WriteFile(queriesPath, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelsPath, []byte("models:\n  - name: expensive\n    max_tokens: 8193\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{}); err == nil {
		t.Fatal("model token amplification unexpectedly accepted")
	}
}

func TestBenchmarkCombinedWorkBudgetBoundaries(t *testing.T) {
	runner, _ := newPipelineSecurityTestRunner(t, "")
	_, modelsPath, queriesPath := writePipelineTestInputs(t, runner)

	var queries strings.Builder
	for range maxBenchmarkQueries {
		queries.WriteString("{}\n")
	}
	if err := os.WriteFile(queriesPath, []byte(queries.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(modelsPath, []byte("models:\n  - name: boundary\n    max_tokens: 1000\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{}); err != nil {
		t.Fatalf("exact task and token budgets rejected: %v", err)
	}

	if err := os.WriteFile(modelsPath, []byte("models:\n  - name: first\n    max_tokens: 1\n  - name: second\n    max_tokens: 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{Limit: 10_001}); err == nil {
		t.Fatal("workload above the task budget unexpectedly accepted")
	}

	if err := os.WriteFile(modelsPath, []byte("models:\n  - name: token-heavy\n    max_tokens: 1001\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBenchmarkWorkload(modelsPath, queriesPath, BenchmarkRequest{}); err == nil {
		t.Fatal("workload above the declared output token budget unexpectedly accepted")
	}
}

func TestPipelineJobAdmissionIsBoundedAndReleaseIsIdempotent(t *testing.T) {
	runner, _ := newPipelineSecurityTestRunner(t, "")
	releaseOne, err := runner.acquireJobSlot(pipelineBenchmarkJob)
	if err != nil {
		t.Fatal(err)
	}
	releaseTwo, err := runner.acquireJobSlot(pipelineBenchmarkJob)
	if err != nil {
		t.Fatal(err)
	}
	if _, capacityErr := runner.acquireJobSlot(pipelineBenchmarkJob); !errors.Is(capacityErr, ErrJobCapacityExceeded) {
		t.Fatalf("third benchmark admission error = %v", capacityErr)
	}
	releaseTrain, err := runner.acquireJobSlot(pipelineTrainingJob)
	if err != nil {
		t.Fatal(err)
	}
	if _, capacityErr := runner.acquireJobSlot(pipelineTrainingJob); !errors.Is(capacityErr, ErrJobCapacityExceeded) {
		t.Fatalf("total/training admission error = %v", capacityErr)
	}
	releaseOne()
	releaseOne()
	releaseTwo()
	releaseTrain()

	runner.admissionMu.Lock()
	active := runner.activeJobs
	runner.admissionMu.Unlock()
	if active != 0 {
		t.Fatalf("active jobs after release = %d", active)
	}
	resumedRelease, err := runner.acquireJobSlot(pipelineBenchmarkJob)
	if err != nil {
		t.Fatalf("capacity did not recover: %v", err)
	}
	resumedRelease()
}

func TestPipelineJobPanicFailsJobAndReleasesAdmission(t *testing.T) {
	runner, _ := newPipelineSecurityTestRunner(t, "")
	job, err := runner.createJob("benchmark")
	if err != nil {
		t.Fatal(err)
	}
	release, err := runner.acquireJobSlot(pipelineBenchmarkJob)
	if err != nil {
		t.Fatal(err)
	}
	runner.startAsyncJob(job.ID, release, func() { panic("sensitive panic payload") })
	failed := waitForPipelineJob(t, runner, job.ID, StatusFailed)
	if strings.Contains(failed.Error, "sensitive panic payload") {
		t.Fatalf("panic content persisted: %+v", failed)
	}
	runner.admissionMu.Lock()
	active := runner.activeJobs
	runner.admissionMu.Unlock()
	if active != 0 {
		t.Fatalf("panic leaked admission slot: %d active", active)
	}
}

func TestNewRunnerRemovesStaleUploadsWithoutFollowingSymlinks(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "pipeline-data")
	uploadRoot := filepath.Join(dataDir, managedUploadDirectory)
	staleDir := filepath.Join(uploadRoot, "stale-job")
	if err := os.MkdirAll(staleDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staleDir, "input"), []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}
	outsideDir := filepath.Join(root, "outside")
	if err := os.MkdirAll(outsideDir, 0o700); err != nil {
		t.Fatal(err)
	}
	outsideFile := filepath.Join(outsideDir, "keep")
	if err := os.WriteFile(outsideFile, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outsideDir, filepath.Join(uploadRoot, "outside-link")); err != nil {
		t.Fatal(err)
	}
	wf, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer wf.Close()
	if _, runnerErr := NewRunner(RunnerConfig{DataDir: dataDir, Workflow: wf}); runnerErr != nil {
		t.Fatal(runnerErr)
	}
	entries, err := os.ReadDir(uploadRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("stale upload entries remain: %v", entries)
	}
	if content, err := os.ReadFile(outsideFile); err != nil || string(content) != "keep" {
		t.Fatalf("staging cleanup followed symlink: content=%q err=%v", content, err)
	}
}

func TestMLServiceSSEIsBoundedStrictAndContentFree(t *testing.T) {
	runner, _ := newPipelineSecurityTestRunner(t, "")
	job, err := runner.createJob("benchmark")
	if err != nil {
		t.Fatal(err)
	}

	secret := "sidecar-secret-value"
	failureEvent := fmt.Sprintf("data: {\"percent\":100,\"step\":\"%s\",\"message\":\"%s\",\"done\":true,\"success\":false}\n\n", secret, secret)
	_, err = runner.readSSEStream(job.ID, io.NopCloser(strings.NewReader(failureEvent)))
	if err == nil || strings.Contains(err.Error(), secret) || err.Error() != errMLServiceJobFailed.Error() {
		t.Fatalf("failure error = %v", err)
	}
	events, err := runner.ListProgressEvents(job.ID, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || strings.Contains(events[0].Step, secret) || strings.Contains(events[0].Message, secret) {
		t.Fatalf("raw sidecar content persisted: %+v", events)
	}

	malformed := "data: {\"percent\":1,\"step\":\"ok\",\"message\":\"" + secret + "\",\"unknown\":true}\n\n"
	_, err = runner.readSSEStream(job.ID, io.NopCloser(strings.NewReader(malformed)))
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatalf("malformed event error = %v", err)
	}

	longLine := "data: " + strings.Repeat("x", mlSSEMaximumLineBytes) + "\n"
	if _, streamErr := runner.readSSEStream(job.ID, io.NopCloser(strings.NewReader(longLine))); streamErr == nil {
		t.Fatal("oversized SSE line unexpectedly accepted")
	}

	output := filepath.Join(runner.dataDir, "job-output.jsonl")
	successData, _ := json.Marshal(sseEvent{
		Percent: 100, Step: "done", Message: "done", Done: true, Success: true, OutputFiles: []string{output},
	})
	files, err := runner.readSSEStream(job.ID, io.NopCloser(bytes.NewReader(append(append([]byte("data: "), successData...), '\n', '\n'))))
	if err != nil || len(files) != 1 || files[0] != output {
		t.Fatalf("success event = files %v, err %v", files, err)
	}
}

func TestHTTPPipelineAcceptsOnlyManagedOutputsAndCleansUploads(t *testing.T) {
	t.Run("managed output completes", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer r.Body.Close()
			var payload struct {
				OutputDir string `json:"output_dir"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Errorf("decode request: %v", err)
				http.Error(w, "bad", http.StatusBadRequest)
				return
			}
			output := filepath.Join(payload.OutputDir, "benchmark_output.jsonl")
			if err := os.WriteFile(output, []byte("{}\n"), 0o600); err != nil {
				t.Errorf("write output: %v", err)
				http.Error(w, "bad", http.StatusInternalServerError)
				return
			}
			data, _ := json.Marshal(sseEvent{Percent: 100, Step: "done", Message: "ok", Done: true, Success: true, OutputFiles: []string{output}})
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		}))
		defer server.Close()

		runner, dataDir := newPipelineSecurityTestRunner(t, server.URL)
		uploadDir, modelsPath, queriesPath := writePipelineTestInputs(t, runner)
		jobID, err := runner.RunBenchmark(context.Background(), modelsPath, queriesPath, BenchmarkRequest{})
		if err != nil {
			t.Fatal(err)
		}
		job := waitForPipelineJob(t, runner, jobID, StatusCompleted)
		if len(job.OutputFiles) != 1 || !pathWithinRoot(dataDir, job.OutputFiles[0]) {
			t.Fatalf("unexpected output files: %v", job.OutputFiles)
		}
		waitForPipelineRunnerIdle(t, runner)
		if _, err := os.Stat(uploadDir); !os.IsNotExist(err) {
			t.Fatalf("managed upload was not removed: %v", err)
		}
	})

	t.Run("outside output fails closed", func(t *testing.T) {
		outside := filepath.Join(t.TempDir(), "outside.jsonl")
		if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
			t.Fatal(err)
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			data, _ := json.Marshal(sseEvent{Percent: 100, Step: "done", Message: "secret-message", Done: true, Success: true, OutputFiles: []string{outside}})
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		}))
		defer server.Close()

		runner, _ := newPipelineSecurityTestRunner(t, server.URL)
		_, modelsPath, queriesPath := writePipelineTestInputs(t, runner)
		jobID, err := runner.RunBenchmark(context.Background(), modelsPath, queriesPath, BenchmarkRequest{})
		if err != nil {
			t.Fatal(err)
		}
		job := waitForPipelineJob(t, runner, jobID, StatusFailed)
		if len(job.OutputFiles) != 0 || strings.Contains(job.Error, outside) || strings.Contains(job.Error, "secret-message") {
			t.Fatalf("unsafe sidecar result persisted: %+v", job)
		}
	})

	t.Run("error body is never persisted", func(t *testing.T) {
		const secretBody = "response-body-secret"
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, secretBody, http.StatusInternalServerError)
		}))
		defer server.Close()

		runner, _ := newPipelineSecurityTestRunner(t, server.URL)
		_, modelsPath, queriesPath := writePipelineTestInputs(t, runner)
		jobID, err := runner.RunBenchmark(context.Background(), modelsPath, queriesPath, BenchmarkRequest{})
		if err != nil {
			t.Fatal(err)
		}
		job := waitForPipelineJob(t, runner, jobID, StatusFailed)
		if strings.Contains(job.Error, secretBody) {
			t.Fatalf("response body persisted: %+v", job)
		}
	})
}

func TestSequentialTrainJobsKeepHistoricalSnapshots(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload struct {
			OutputDir string `json:"output_dir"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		version := requests.Add(1)
		output := filepath.Join(payload.OutputDir, "knn_model.json")
		content := fmt.Sprintf("{\"version\":%d}\n", version)
		if err := os.WriteFile(output, []byte(content), 0o644); err != nil {
			http.Error(w, "write failed", http.StatusInternalServerError)
			return
		}
		data, _ := json.Marshal(sseEvent{
			Percent: 100, Step: "done", Message: "ok", Done: true, Success: true,
			OutputFiles: []string{output},
		})
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	}))
	defer server.Close()

	runner, _ := newPipelineSecurityTestRunner(t, server.URL)
	runTraining := func() (string, *Job) {
		uploadDir, err := runner.CreateUploadDir("training-")
		if err != nil {
			t.Fatal(err)
		}
		input := filepath.Join(uploadDir, "training.jsonl")
		if writeErr := os.WriteFile(input, []byte("{\"query\":\"hello\"}\n"), 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
		jobID, err := runner.RunTrain(context.Background(), input, TrainRequest{Algorithms: []string{"knn"}})
		if err != nil {
			t.Fatal(err)
		}
		job := waitForPipelineJob(t, runner, jobID, StatusCompleted)
		waitForPipelineRunnerIdle(t, runner)
		if _, err := os.Stat(uploadDir); !os.IsNotExist(err) {
			t.Fatalf("training upload was not removed: %v", err)
		}
		if len(job.OutputFiles) != 1 || filepath.Dir(job.OutputFiles[0]) != runner.JobDir(jobID) {
			t.Fatalf("job output is not bound to its immutable directory: %v", job.OutputFiles)
		}
		return jobID, job
	}

	readOutput := func(jobID string, job *Job) string {
		file, info, err := runner.OpenJobOutputFile(jobID, "train", job.OutputFiles[0])
		if err != nil {
			t.Fatalf("open historical job output: %v", err)
		}
		defer file.Close()
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("snapshot mode = %o, want 600", info.Mode().Perm())
		}
		content, err := io.ReadAll(io.LimitReader(file, 1024))
		if err != nil {
			t.Fatal(err)
		}
		return string(content)
	}

	firstID, firstJob := runTraining()
	if got := readOutput(firstID, firstJob); got != "{\"version\":1}\n" {
		t.Fatalf("first snapshot = %q", got)
	}
	// The deployment-facing stable path is mutable by design. Historical job
	// artifacts must be independent copies rather than hard links to it.
	if err := os.WriteFile(filepath.Join(runner.TrainDir(), "knn_model.json"), []byte("{\"version\":99}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := readOutput(firstID, firstJob); got != "{\"version\":1}\n" {
		t.Fatalf("historical snapshot changed after in-place stable output mutation: %q", got)
	}
	secondID, secondJob := runTraining()
	if got := readOutput(secondID, secondJob); got != "{\"version\":2}\n" {
		t.Fatalf("second snapshot = %q", got)
	}
	if got := readOutput(firstID, firstJob); got != "{\"version\":1}\n" {
		t.Fatalf("historical snapshot changed after the next training run: %q", got)
	}
	if file, _, err := runner.OpenJobOutputFile(firstID, "train", secondJob.OutputFiles[0]); err == nil {
		_ = file.Close()
		t.Fatal("a later training job output was accepted for the historical job")
	}
	stableContent, err := os.ReadFile(filepath.Join(runner.TrainDir(), "knn_model.json"))
	if err != nil || string(stableContent) != "{\"version\":2}\n" {
		t.Fatalf("stable deployment output = %q, err %v", stableContent, err)
	}
}

func TestMLServiceURLAndClientAreFailClosed(t *testing.T) {
	for _, rawURL := range []string{
		"ftp://localhost:8686", "http://user:pass@localhost:8686", "http://localhost:8686?token=secret", "//localhost:8686",
	} {
		if _, err := validateMLServiceURL(rawURL); err == nil {
			t.Fatalf("unsafe ML service URL accepted: %q", rawURL)
		}
	}
	client := newMLServiceHTTPClient()
	if client.Timeout <= 0 {
		t.Fatal("ML service client has no total timeout")
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok || transport.TLSClientConfig == nil || transport.TLSClientConfig.MinVersion < tls.VersionTLS12 || transport.Proxy != nil {
		t.Fatalf("unsafe ML service transport: %#v", client.Transport)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "/target", http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	resp, err := client.Get(server.URL + "/redirect")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("redirect followed: status=%d", resp.StatusCode)
	}
}
