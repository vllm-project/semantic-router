package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func newMLPipelineHandlerTestRunner(t *testing.T) (*mlpipeline.Runner, *workflowstore.Store, string) {
	return newMLPipelineHandlerTestRunnerWithService(t, "")
}

func newMLPipelineHandlerTestRunnerWithService(t *testing.T, serviceURL string) (*mlpipeline.Runner, *workflowstore.Store, string) {
	t.Helper()
	root := t.TempDir()
	store, err := workflowstore.Open(filepath.Join(root, "workflow.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatalf("open workflow store: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	dataDir := filepath.Join(root, "pipeline-data")
	runner, err := mlpipeline.NewRunner(mlpipeline.RunnerConfig{
		DataDir:      dataDir,
		TrainingDir:  t.TempDir(),
		PythonPath:   "python3",
		MLServiceURL: serviceURL,
		Workflow:     store,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	return runner, store, dataDir
}

func multipartRequest(t *testing.T, targetURL, fieldName, filename string, content []byte, fields map[string]string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	for name, value := range fields {
		if err := writer.WriteField(name, value); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, targetURL, bytes.NewReader(body.Bytes()))
	request.Header.Set("Content-Type", writer.FormDataContentType())
	return request
}

func TestMLPipelineUploadIgnoresFilenameAndUsesPrivateBoundedFile(t *testing.T) {
	targetDir := t.TempDir()
	request := multipartRequest(t, "/upload", "training_data", "../../escape.jsonl", []byte("safe"), nil)
	if err := request.ParseMultipartForm(1); err != nil {
		t.Fatal(err)
	}
	defer cleanupMultipartForm(request)

	path, err := saveUploadedFile(request, "training_data", targetDir, "training.jsonl", 16)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(targetDir, "training.jsonl") {
		t.Fatalf("saved path = %q", path)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("uploaded file mode = %o, want 600", got)
	}
	content, err := os.ReadFile(path)
	if err != nil || string(content) != "safe" {
		t.Fatalf("uploaded content = %q, err=%v", content, err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(targetDir), "escape.jsonl")); !os.IsNotExist(err) {
		t.Fatalf("caller filename escaped target directory: %v", err)
	}

	oversizedRequest := multipartRequest(t, "/upload", "training_data", "large.jsonl", []byte("12345"), nil)
	if err := oversizedRequest.ParseMultipartForm(1); err != nil {
		t.Fatal(err)
	}
	defer cleanupMultipartForm(oversizedRequest)
	if _, err := saveUploadedFile(oversizedRequest, "training_data", t.TempDir(), "training.jsonl", 4); err == nil {
		t.Fatal("oversized uploaded file unexpectedly accepted")
	}
}

func TestMLPipelineMultipartTransportLimitAndCleanup(t *testing.T) {
	request := multipartRequest(t, "/upload", "file", "large.bin", bytes.Repeat([]byte("x"), 1024), nil)
	status, err := parseBoundedMultipart(httptest.NewRecorder(), request, 128, 1)
	if err == nil || status != http.StatusRequestEntityTooLarge {
		t.Fatalf("parseBoundedMultipart = status %d, err %v", status, err)
	}
	cleanupMultipartForm(request)

	tempRoot := t.TempDir()
	t.Setenv("TMPDIR", tempRoot)
	request = multipartRequest(t, "/upload", "file", "spooled.bin", bytes.Repeat([]byte("x"), 1024), nil)
	status, err = parseBoundedMultipart(httptest.NewRecorder(), request, 4096, 1)
	if err != nil || status != 0 {
		t.Fatalf("parseBoundedMultipart = status %d, err %v", status, err)
	}
	entriesBefore, err := os.ReadDir(tempRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(entriesBefore) == 0 {
		t.Fatal("multipart test did not spool to disk")
	}
	cleanupMultipartForm(request)
	entriesAfter, err := os.ReadDir(tempRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(entriesAfter) != 0 {
		t.Fatalf("multipart temporary files leaked: %v", entriesAfter)
	}
}

func TestMLPipelineTrainRejectsExternalAndUntrustedBenchmarkPaths(t *testing.T) {
	runner, store, dataDir := newMLPipelineHandlerTestRunner(t)
	handler := NewMLPipelineHandler(runner).RunTrainHandler()

	externalPath := filepath.Join(t.TempDir(), "host-secret.jsonl")
	if err := os.WriteFile(externalPath, []byte("host-secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/train", strings.NewReader(
		`{"benchmark_data_path":"`+externalPath+`","config":{}}`,
	))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest || strings.Contains(recorder.Body.String(), externalPath) {
		t.Fatalf("external path request = status %d, body %q", recorder.Code, recorder.Body.String())
	}

	jobID := "ml-benchmark-untrusted"
	if err := store.PutMLJob(workflowstore.MLJobRecord{
		ID: jobID, Type: "benchmark", Status: string(mlpipeline.StatusCompleted), CreatedAt: time.Now(), OutputFiles: []string{externalPath}, Progress: 100,
	}); err != nil {
		t.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/train", strings.NewReader(
		`{"benchmark_job_id":"`+jobID+`","config":{}}`,
	))
	request.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest || strings.Contains(recorder.Body.String(), externalPath) {
		t.Fatalf("untrusted output request = status %d, body %q", recorder.Code, recorder.Body.String())
	}

	insideTarget := filepath.Join(dataDir, "inside.jsonl")
	if err := os.WriteFile(insideTarget, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	symlink := filepath.Join(dataDir, "inside-link.jsonl")
	if err := os.Symlink(insideTarget, symlink); err != nil {
		t.Fatal(err)
	}
	jobID = "ml-benchmark-symlink"
	if err := store.PutMLJob(workflowstore.MLJobRecord{
		ID: jobID, Type: "benchmark", Status: string(mlpipeline.StatusCompleted), CreatedAt: time.Now(), OutputFiles: []string{symlink}, Progress: 100,
	}); err != nil {
		t.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/train", strings.NewReader(
		`{"benchmark_job_id":"`+jobID+`","config":{}}`,
	))
	request.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("symlink output request status = %d, body %q", recorder.Code, recorder.Body.String())
	}
}

func TestMLPipelineDownloadOpensOnlyManagedRegularOutput(t *testing.T) {
	runner, store, _ := newMLPipelineHandlerTestRunner(t)
	handler := NewMLPipelineHandler(runner).DownloadOutputHandler()

	managedJob := "ml-benchmark-managed"
	managedDir := runner.JobDir(managedJob)
	if err := os.MkdirAll(managedDir, 0o700); err != nil {
		t.Fatal(err)
	}
	managedPath := filepath.Join(managedDir, "benchmark_output.jsonl")
	if err := os.WriteFile(managedPath, []byte("managed-output"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := store.PutMLJob(workflowstore.MLJobRecord{
		ID: managedJob, Type: "benchmark", Status: string(mlpipeline.StatusCompleted), CreatedAt: time.Now(), OutputFiles: []string{managedPath}, Progress: 100,
	}); err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/ml-pipeline/download/"+managedJob+"/0", nil))
	if recorder.Code != http.StatusOK || recorder.Body.String() != "managed-output" {
		t.Fatalf("managed download = status %d, body %q", recorder.Code, recorder.Body.String())
	}
	if disposition := recorder.Header().Get("Content-Disposition"); !strings.Contains(disposition, "benchmark_output.jsonl") {
		t.Fatalf("Content-Disposition = %q", disposition)
	}

	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/ml-pipeline/download/"+managedJob+"/-1", nil))
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("negative index status = %d", recorder.Code)
	}

	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("do-not-serve"), 0o600); err != nil {
		t.Fatal(err)
	}
	unsafeJob := "ml-benchmark-outside"
	if err := store.PutMLJob(workflowstore.MLJobRecord{
		ID: unsafeJob, Type: "benchmark", Status: string(mlpipeline.StatusCompleted), CreatedAt: time.Now(), OutputFiles: []string{outside}, Progress: 100,
	}); err != nil {
		t.Fatal(err)
	}
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/ml-pipeline/download/"+unsafeJob+"/0", nil))
	if recorder.Code != http.StatusNotFound || strings.Contains(recorder.Body.String(), "do-not-serve") {
		t.Fatalf("outside download = status %d, body %q", recorder.Code, recorder.Body.String())
	}
}

func TestMLPipelineBenchmarkCapacityReturns429AndCleansRejectedUpload(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload struct {
			OutputDir string `json:"output_dir"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		started <- struct{}{}
		<-release
		output := filepath.Join(payload.OutputDir, "benchmark_output.jsonl")
		if err := os.WriteFile(output, []byte("{}\n"), 0o600); err != nil {
			http.Error(w, "bad", http.StatusInternalServerError)
			return
		}
		data, _ := json.Marshal(map[string]any{
			"percent": 100, "step": "done", "message": "done", "done": true, "success": true, "output_files": []string{output},
		})
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	}))
	defer server.Close()

	runner, _, dataDir := newMLPipelineHandlerTestRunnerWithService(t, server.URL)
	jobIDs := make([]string, 0, 2)
	for i := 0; i < 2; i++ {
		uploadDir, err := runner.CreateUploadDir("admitted-")
		if err != nil {
			t.Fatal(err)
		}
		modelsPath := filepath.Join(uploadDir, "models.yaml")
		queriesPath := filepath.Join(uploadDir, "queries.jsonl")
		if writeModelsErr := os.WriteFile(modelsPath, []byte("models:\n  - name: model-a\n"), 0o600); writeModelsErr != nil {
			t.Fatal(writeModelsErr)
		}
		if writeQueriesErr := os.WriteFile(queriesPath, []byte("{\"query\":\"hello\"}\n"), 0o600); writeQueriesErr != nil {
			t.Fatal(writeQueriesErr)
		}
		jobID, err := runner.RunBenchmark(context.Background(), modelsPath, queriesPath, mlpipeline.BenchmarkRequest{})
		if err != nil {
			t.Fatal(err)
		}
		jobIDs = append(jobIDs, jobID)
	}
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(3 * time.Second):
			t.Fatal("admitted benchmark did not reach sidecar")
		}
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	modelsPart, _ := writer.CreateFormFile("models_yaml", "../../models.yaml")
	_, _ = modelsPart.Write([]byte("models:\n  - name: model-a\n"))
	queriesPart, _ := writer.CreateFormFile("queries_jsonl", "../../queries.jsonl")
	_, _ = queriesPart.Write([]byte("{\"query\":\"hello\"}\n"))
	_ = writer.WriteField("config", `{}`)
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/benchmark", bytes.NewReader(body.Bytes()))
	request.Header.Set("Content-Type", writer.FormDataContentType())
	recorder := httptest.NewRecorder()
	NewMLPipelineHandler(runner).RunBenchmarkHandler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusTooManyRequests || recorder.Header().Get("Retry-After") == "" {
		t.Fatalf("capacity response = status %d headers=%v body=%q", recorder.Code, recorder.Header(), recorder.Body.String())
	}
	uploadEntries, err := os.ReadDir(filepath.Join(dataDir, ".uploads"))
	if err != nil {
		t.Fatal(err)
	}
	if len(uploadEntries) != 2 {
		// Only the two admitted uploads remain until their jobs finish.
		t.Fatalf("rejected upload leaked; entries=%v", uploadEntries)
	}

	close(release)
	deadline := time.Now().Add(5 * time.Second)
	for _, jobID := range jobIDs {
		for time.Now().Before(deadline) {
			job := runner.GetJob(jobID)
			if job != nil && job.Status == mlpipeline.StatusCompleted {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if job := runner.GetJob(jobID); job == nil || job.Status != mlpipeline.StatusCompleted {
			t.Fatalf("job %s did not complete: %+v", jobID, job)
		}
	}
}

func TestMLPipelineConfigRequestIsStrictAndBounded(t *testing.T) {
	runner, _, _ := newMLPipelineHandlerTestRunner(t)
	handler := NewMLPipelineHandler(runner).GenerateConfigHandler()

	tests := []struct {
		name string
		body io.Reader
		want int
	}{
		{name: "unknown field", body: strings.NewReader(`{"unknown":true}`), want: http.StatusBadRequest},
		{name: "trailing JSON", body: strings.NewReader(`{} {}`), want: http.StatusBadRequest},
		{name: "oversized", body: strings.NewReader(`{"models_path":"` + strings.Repeat("x", mlJSONBodyLimit) + `"}`), want: http.StatusRequestEntityTooLarge},
		{name: "invalid decision", body: strings.NewReader(`{"decisions":[{"name":"bad\nname","algorithm":"knn"}]}`), want: http.StatusBadRequest},
		{name: "valid", body: strings.NewReader(`{"models_path":"/data/ml-pipeline/ml-train","device":"cpu","decisions":[{"name":"default","domains":["general"],"algorithm":"knn","priority":1,"model_names":["model-a"]}]}`), want: http.StatusCreated},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/config", test.body)
			request.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != test.want {
				t.Fatalf("status = %d, body %q; want %d", recorder.Code, recorder.Body.String(), test.want)
			}
		})
	}
}

func TestMLPipelineSSERegistryIsBoundedAndReclaimed(t *testing.T) {
	runner, _, _ := newMLPipelineHandlerTestRunner(t)
	handler := NewMLPipelineHandler(runner)

	registrations := make([]struct {
		id string
		ch chan mlpipeline.ProgressUpdate
	}, 0, mlSSEClientsPerJobLimit)
	for i := 0; i < mlSSEClientsPerJobLimit; i++ {
		id, ch, _, _, err := handler.registerSSEClient("job-one")
		if err != nil {
			t.Fatalf("register client %d: %v", i, err)
		}
		registrations = append(registrations, struct {
			id string
			ch chan mlpipeline.ProgressUpdate
		}{id: id, ch: ch})
	}
	if _, _, _, _, err := handler.registerSSEClient("job-one"); err == nil {
		t.Fatal("per-job SSE capacity was not enforced")
	}
	handler.broadcastProgress(mlpipeline.ProgressUpdate{JobID: "job-one", Percent: 50, Step: "running", Message: "progress"})
	select {
	case update := <-registrations[0].ch:
		if update.Percent != 50 {
			t.Fatalf("broadcast update = %+v", update)
		}
	case <-time.After(time.Second):
		t.Fatal("registered client did not receive broadcast")
	}
	for len(registrations[0].ch) < cap(registrations[0].ch) {
		registrations[0].ch <- mlpipeline.ProgressUpdate{JobID: "job-one", Percent: 60}
	}
	handler.broadcastProgress(mlpipeline.ProgressUpdate{JobID: "job-one", Percent: 100, Step: "done", Message: "done"})
	foundTerminal := false
	for len(registrations[0].ch) > 0 {
		if (<-registrations[0].ch).Percent >= 100 {
			foundTerminal = true
		}
	}
	if !foundTerminal {
		t.Fatal("terminal progress was dropped from a full client queue")
	}
	for _, registration := range registrations {
		handler.unregisterSSEClient("job-one", registration.id)
	}
	handler.sseMu.Lock()
	_, outerEntryExists := handler.sseClients["job-one"]
	clientCount := handler.sseClientCount
	handler.sseMu.Unlock()
	if outerEntryExists || clientCount != 0 {
		t.Fatalf("SSE registry leaked: outer=%v clients=%d", outerEntryExists, clientCount)
	}

	for i := 0; i < mlTerminalProgressRetention+100; i++ {
		handler.broadcastProgress(mlpipeline.ProgressUpdate{
			JobID: fmt.Sprintf("terminal-%d", i), Percent: 100, Step: "done", Message: "done",
		})
	}
	handler.sseMu.Lock()
	retainedProgress := len(handler.lastProgress)
	retainedTerminal := len(handler.terminalOrder)
	handler.sseMu.Unlock()
	if retainedProgress > mlTerminalProgressRetention || retainedTerminal > mlTerminalProgressRetention {
		t.Fatalf("terminal progress retention is unbounded: progress=%d order=%d", retainedProgress, retainedTerminal)
	}
}

func TestMLPipelineSSEConcurrentSubscribeBroadcastUnsubscribe(t *testing.T) {
	runner, _, _ := newMLPipelineHandlerTestRunner(t)
	handler := NewMLPipelineHandler(runner)
	var wg sync.WaitGroup
	for i := 0; i < 128; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			jobID := fmt.Sprintf("race-job-%d", i%8)
			clientID, _, _, _, err := handler.registerSSEClient(jobID)
			if err != nil {
				return
			}
			for percent := 1; percent <= 100; percent += 33 {
				handler.broadcastProgress(mlpipeline.ProgressUpdate{
					JobID: jobID, Percent: percent, Step: "race", Message: "bounded",
				})
			}
			handler.unregisterSSEClient(jobID, clientID)
		}(i)
	}
	wg.Wait()
	handler.sseMu.Lock()
	clientCount := handler.sseClientCount
	outerEntries := len(handler.sseClients)
	handler.sseMu.Unlock()
	if clientCount != 0 || outerEntries != 0 {
		t.Fatalf("concurrent SSE registry leaked: clients=%d jobs=%d", clientCount, outerEntries)
	}
}
