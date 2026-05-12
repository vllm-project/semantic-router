package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

type testEvaluationHarness struct {
	handler *EvaluationHandler
	db      *evaluation.DB
	rootDir string
}

func newTestEvaluationHarness(t *testing.T) *testEvaluationHarness {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "eval.db")
	evalDB, err := evaluation.NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	t.Cleanup(func() { _ = evalDB.Close() })
	runner := evaluation.NewRunner(evaluation.RunnerConfig{
		DB:          evalDB,
		ProjectRoot: dir,
		ResultsDir:  filepath.Join(dir, "results"),
	})
	return &testEvaluationHarness{
		handler: NewEvaluationHandler(evalDB, runner, false, "http://router:8080", "http://envoy:8801"),
		db:      evalDB,
		rootDir: dir,
	}
}

func newTestEvaluationHandler(t *testing.T) *EvaluationHandler {
	t.Helper()
	return newTestEvaluationHarness(t).handler
}

func TestCreateTaskHandler_MoMWithAccuracyReturns201(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "system-accuracy",
		"config": map[string]interface{}{
			"level":       "mom",
			"dimensions":  []string{"accuracy"},
			"endpoint":    "http://localhost:8801",
			"max_samples": 10,
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want %d; body = %s", rec.Code, http.StatusCreated, rec.Body.String())
	}
	var task models.EvaluationTask
	if err := json.NewDecoder(rec.Body).Decode(&task); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if task.Config.Level != models.LevelMoM || len(task.Config.Dimensions) != 1 || task.Config.Dimensions[0] != models.DimensionAccuracy {
		t.Errorf("task config level=%q dimensions=%v", task.Config.Level, task.Config.Dimensions)
	}
}

func TestCreateTaskHandler_MoMWithDomainReturns400(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "invalid-mom",
		"config": map[string]interface{}{
			"level":      "mom",
			"dimensions": []string{"domain"},
			"endpoint":   "http://localhost:8801",
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d; body = %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}
}

func TestCreateTaskHandler_MoMWithSignalDatasetReturns400(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "invalid-mom-dataset",
		"config": map[string]interface{}{
			"level":      "mom",
			"dimensions": []string{"accuracy"},
			"datasets": map[string]interface{}{
				"accuracy": []string{"mmlu-pro-en"},
			},
			"endpoint": "http://localhost:8801",
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d; body = %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}
}

func TestCreateTaskHandler_RouterWithDomainReturns201(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "signal-eval",
		"config": map[string]interface{}{
			"level":       "router",
			"dimensions":  []string{"domain"},
			"endpoint":    "http://localhost:8080/api/v1/eval",
			"max_samples": 20,
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want %d; body = %s", rec.Code, http.StatusCreated, rec.Body.String())
	}
}

func TestCreateTaskHandler_RouterPlaceholderEndpointReplacedByConfiguredOverride(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "signal-eval-override",
		"config": map[string]interface{}{
			"level":      "router",
			"dimensions": []string{"domain"},
			// Frontend always sends this hardcoded placeholder; the backend must
			// substitute the configured TARGET_ROUTER_API_URL when present so the
			// dashboard can reach the router across pods.
			"endpoint":    "http://localhost:8080/api/v1/eval",
			"max_samples": 5,
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want %d; body = %s", rec.Code, http.StatusCreated, rec.Body.String())
	}

	var task models.EvaluationTask
	if err := json.NewDecoder(rec.Body).Decode(&task); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	want := "http://router:8080/api/v1/eval"
	if task.Config.Endpoint != want {
		t.Errorf("endpoint = %q, want %q (router override should replace placeholder)", task.Config.Endpoint, want)
	}
}

func TestCreateTaskHandler_RouterExplicitEndpointPreserved(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	custom := "http://my-router.example.com:9090/api/v1/eval"
	body := map[string]interface{}{
		"name": "signal-eval-custom",
		"config": map[string]interface{}{
			"level":       "router",
			"dimensions":  []string{"domain"},
			"endpoint":    custom,
			"max_samples": 5,
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want %d; body = %s", rec.Code, http.StatusCreated, rec.Body.String())
	}

	var task models.EvaluationTask
	if err := json.NewDecoder(rec.Body).Decode(&task); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if task.Config.Endpoint != custom {
		t.Errorf("endpoint = %q, want %q (non-placeholder endpoint must be preserved)", task.Config.Endpoint, custom)
	}
}

func TestCreateTaskHandler_NormalizesDefaultDatasetsAndExtraKeys(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "signal-eval-defaults",
		"config": map[string]interface{}{
			"level":      "router",
			"dimensions": []string{"domain"},
			"datasets": map[string]interface{}{
				"domain":   []string{"default", "default"},
				"accuracy": []string{"mmlu-pro"},
			},
			"endpoint": "http://localhost:8080/api/v1/eval",
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want %d; body = %s", rec.Code, http.StatusCreated, rec.Body.String())
	}

	var task models.EvaluationTask
	if err := json.NewDecoder(rec.Body).Decode(&task); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if _, ok := task.Config.Datasets["accuracy"]; ok {
		t.Fatalf("unexpected dataset key 'accuracy' in normalized config: %v", task.Config.Datasets)
	}
	if datasets := task.Config.Datasets["domain"]; len(datasets) != 0 {
		t.Fatalf("expected empty normalized domain datasets, got %v", datasets)
	}
}

func TestCreateTaskHandler_MissingNameReturns400(t *testing.T) {
	t.Parallel()
	handler := newTestEvaluationHandler(t)
	body := map[string]interface{}{
		"name": "",
		"config": map[string]interface{}{
			"level":      "router",
			"dimensions": []string{"domain"},
		},
	}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.CreateTaskHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestGetDatasetsHandler_ReturnsDatasets(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/evaluation/datasets", nil)
	GetDatasetsHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d; body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	var out map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out["domain"] == nil || out["accuracy"] == nil {
		t.Errorf("expected domain and accuracy keys in datasets; got %v", out)
	}
}

func TestRunTaskHandlerMarksRerunTaskRunningBeforeBackgroundExecution(t *testing.T) {
	harness := newTestEvaluationHarness(t)

	scriptPath := filepath.Join(harness.rootDir, "src", "training", "model_eval", "mmlu_pro_vllm_eval.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("MkdirAll(script dir): %v", err)
	}
	if err := os.WriteFile(scriptPath, []byte("import time\ntime.sleep(1)\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(script): %v", err)
	}

	task := &models.EvaluationTask{
		Name: "rerun-system-eval",
		Config: models.EvaluationConfig{
			Level:         models.LevelMoM,
			Dimensions:    []models.EvaluationDimension{models.DimensionAccuracy},
			Endpoint:      "http://localhost:8801",
			SamplesPerCat: 1,
		},
	}
	if err := harness.db.CreateTask(task); err != nil {
		t.Fatalf("CreateTask(): %v", err)
	}
	if err := harness.db.UpdateTaskStatus(task.ID, models.StatusFailed, "previous failure"); err != nil {
		t.Fatalf("UpdateTaskStatus(failed): %v", err)
	}
	if err := harness.db.UpdateTaskProgress(task.ID, 100, "Failed"); err != nil {
		t.Fatalf("UpdateTaskProgress(): %v", err)
	}

	bodyBytes, _ := json.Marshal(map[string]string{"task_id": task.ID})
	req := httptest.NewRequest(http.MethodPost, "/api/evaluation/run", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	harness.handler.RunTaskHandler().ServeHTTP(rec, req)
	t.Cleanup(func() {
		_ = harness.handler.runner.CancelTask(task.ID)
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}

	updatedTask, err := harness.db.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask(): %v", err)
	}
	if updatedTask == nil {
		t.Fatal("expected task to exist")
	}
	if updatedTask.Status != models.StatusRunning {
		t.Fatalf("task status = %s, want %s", updatedTask.Status, models.StatusRunning)
	}
	if updatedTask.ProgressPercent != 0 {
		t.Fatalf("task progress = %d, want 0", updatedTask.ProgressPercent)
	}
	if updatedTask.CurrentStep != "Starting evaluation" {
		t.Fatalf("task current_step = %q, want %q", updatedTask.CurrentStep, "Starting evaluation")
	}
}
