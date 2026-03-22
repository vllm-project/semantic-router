package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func TestCreateTaskHandler_ValidatesDimensionLevel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "eval.db")
	evalDB, err := evaluation.NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer evalDB.Close()

	runner := evaluation.NewRunner(evaluation.RunnerConfig{
		DB:          evalDB,
		ProjectRoot: dir,
		ResultsDir:  filepath.Join(dir, "results"),
	})
	handler := NewEvaluationHandler(evalDB, runner, false, "http://router:8080", "http://envoy:8801")

	t.Run("mom with accuracy dimension returns 201", func(t *testing.T) {
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
	})

	t.Run("mom with domain dimension returns 400", func(t *testing.T) {
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
	})

	t.Run("router with domain dimension returns 201", func(t *testing.T) {
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
	})

	t.Run("missing name returns 400", func(t *testing.T) {
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
	})
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
