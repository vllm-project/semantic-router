package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

type registeredEvaluationSSEClient struct {
	taskID   string
	clientID string
	client   *evaluationSSEClient
}

func TestEvaluationSSEPerTaskAndGlobalLimits(t *testing.T) {
	handler := &EvaluationHandler{sseClients: make(map[string]*evaluationSSETaskRegistry)}
	taskID := uuid.NewString()
	var perTask []registeredEvaluationSSEClient
	for i := 0; i < evaluationMaxSSEClientsPerTask; i++ {
		clientID, client, err := handler.registerEvaluationSSEClient(taskID)
		if err != nil {
			t.Fatalf("register per-task client %d: %v", i, err)
		}
		perTask = append(perTask, registeredEvaluationSSEClient{taskID, clientID, client})
	}
	if _, _, err := handler.registerEvaluationSSEClient(taskID); !errors.Is(err, errEvaluationSSETaskLimit) {
		t.Fatalf("per-task overflow error = %v, want %v", err, errEvaluationSSETaskLimit)
	}
	for _, registered := range perTask {
		handler.unregisterEvaluationSSEClient(registered.taskID, registered.clientID, registered.client)
	}

	var global []registeredEvaluationSSEClient
	for i := 0; i < evaluationMaxSSEClientsGlobal; i++ {
		globalTaskID := uuid.NewString()
		clientID, client, err := handler.registerEvaluationSSEClient(globalTaskID)
		if err != nil {
			t.Fatalf("register global client %d: %v", i, err)
		}
		global = append(global, registeredEvaluationSSEClient{globalTaskID, clientID, client})
	}
	if _, _, err := handler.registerEvaluationSSEClient(uuid.NewString()); !errors.Is(err, errEvaluationSSEGlobalLimit) {
		t.Fatalf("global overflow error = %v, want %v", err, errEvaluationSSEGlobalLimit)
	}
	for _, registered := range global {
		handler.unregisterEvaluationSSEClient(registered.taskID, registered.clientID, registered.client)
	}
	if handler.sseTotal != 0 || len(handler.sseClients) != 0 {
		t.Fatalf("SSE registry leaked after cleanup: total=%d tasks=%d", handler.sseTotal, len(handler.sseClients))
	}
}

func TestEvaluationSSEConcurrentSubscribeUnsubscribeBroadcastAndFinish(t *testing.T) {
	handler := &EvaluationHandler{sseClients: make(map[string]*evaluationSSETaskRegistry)}
	taskIDs := make([]string, 8)
	for i := range taskIDs {
		taskIDs[i] = uuid.NewString()
	}

	var workers sync.WaitGroup
	for worker := 0; worker < 32; worker++ {
		workers.Add(1)
		go func(worker int) {
			defer workers.Done()
			for iteration := 0; iteration < 100; iteration++ {
				taskID := taskIDs[(worker+iteration)%len(taskIDs)]
				clientID, client, err := handler.registerEvaluationSSEClient(taskID)
				if err != nil {
					continue
				}
				handler.broadcastEvaluationProgress(models.ProgressUpdate{TaskID: taskID, ProgressPercent: iteration % 100})
				if iteration%11 == 0 {
					handler.finishEvaluationSSETask(taskID, models.StatusFailed)
				}
				handler.unregisterEvaluationSSEClient(taskID, clientID, client)
			}
		}(worker)
	}
	workers.Wait()
	for _, taskID := range taskIDs {
		handler.finishEvaluationSSETask(taskID, models.StatusCompleted)
	}

	handler.sseMu.Lock()
	defer handler.sseMu.Unlock()
	if handler.sseTotal != 0 || len(handler.sseClients) != 0 {
		t.Fatalf("SSE registry leaked after concurrent lifecycle: total=%d tasks=%d", handler.sseTotal, len(handler.sseClients))
	}
}

func TestEvaluationRunPanicBecomesFailedAndReleasesSSERegistry(t *testing.T) {
	dir := t.TempDir()
	db, err := evaluation.NewDB(filepath.Join(dir, "eval.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	task := &models.EvaluationTask{
		Name: "panic-boundary",
		Config: models.EvaluationConfig{
			Level:      models.LevelMoM,
			Dimensions: []models.EvaluationDimension{models.DimensionAccuracy},
			Endpoint:   "http://localhost:8801",
		},
	}
	if createErr := db.CreateTask(task); createErr != nil {
		t.Fatal(createErr)
	}
	// The handler DB is valid, while the deliberately nil Runner DB triggers a
	// panic inside the background boundary after admission has succeeded.
	runner := evaluation.NewRunner(evaluation.RunnerConfig{
		DB:          nil,
		ProjectRoot: dir,
		ResultsDir:  filepath.Join(dir, "results"),
	})
	handler := NewEvaluationHandler(db, runner, false, "", "")
	clientID, client, err := handler.registerEvaluationSSEClient(task.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.unregisterEvaluationSSEClient(task.ID, clientID, client)

	body, _ := json.Marshal(models.RunTaskRequest{TaskID: task.ID})
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/evaluation/run", bytes.NewReader(body))
	handler.RunTaskHandler()(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("run status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}

	select {
	case status := <-client.terminal:
		if status != models.StatusFailed {
			t.Fatalf("terminal status = %s, want failed", status)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for panic terminal status")
	}
	updated, err := db.GetTask(task.ID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Status != models.StatusFailed || updated.ErrorMessage != "Evaluation failed" {
		t.Fatalf("task after panic = status %s error %q", updated.Status, updated.ErrorMessage)
	}
	handler.sseMu.Lock()
	defer handler.sseMu.Unlock()
	if handler.sseTotal != 0 || len(handler.sseClients) != 0 {
		t.Fatalf("panic leaked SSE registry: total=%d tasks=%d", handler.sseTotal, len(handler.sseClients))
	}
}

func TestEvaluationHandlersRejectUnsafeIdentifiersAndUnboundedHistory(t *testing.T) {
	handler := newTestEvaluationHandler(t)
	tests := []struct {
		name    string
		handler http.HandlerFunc
		method  string
		target  string
	}{
		{"task traversal", handler.GetTaskHandler(), http.MethodGet, "/api/evaluation/tasks/../../sentinel"},
		{"stream traversal", handler.StreamProgressHandler(), http.MethodGet, "/api/evaluation/stream/../../sentinel"},
		{"history negative limit", handler.GetHistoryHandler(), http.MethodGet, "/api/evaluation/history?metric=accuracy&limit=-1"},
		{"history partial integer", handler.GetHistoryHandler(), http.MethodGet, "/api/evaluation/history?metric=accuracy&limit=12junk"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(test.method, test.target, nil)
			test.handler(recorder, request)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
		})
	}
}

func TestEvaluationCreateRejectsResourceAmplification(t *testing.T) {
	handler := newTestEvaluationHandler(t)
	tests := []map[string]any{
		{"name": "too-concurrent", "config": map[string]any{"level": "mom", "dimensions": []string{"accuracy"}, "concurrent": evaluationMaxConcurrentRequests + 1}},
		{"name": "too-many-samples", "config": map[string]any{"level": "mom", "dimensions": []string{"accuracy"}, "max_samples": evaluationMaxSamples + 1}},
		{"name": "duplicate-dimensions", "config": map[string]any{"level": "mom", "dimensions": []string{"accuracy", "accuracy"}}},
		{"name": "credential-url", "config": map[string]any{"level": "mom", "dimensions": []string{"accuracy"}, "endpoint": "https://user:secret@example.com"}},
	}
	for _, body := range tests {
		raw, _ := json.Marshal(body)
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks", bytes.NewReader(raw))
		handler.CreateTaskHandler()(recorder, request)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("request %q status = %d, want 400; body=%s", body["name"], recorder.Code, recorder.Body.String())
		}
	}
}
