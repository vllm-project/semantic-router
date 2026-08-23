package evaluation

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func TestManagedRunUsesDelegatedAuthorizationWithoutPersistingToken(t *testing.T) {
	dir := t.TempDir()
	evalDB, err := NewDB(filepath.Join(dir, "eval.db"))
	if err != nil {
		t.Fatalf("NewDB() error = %v", err)
	}
	defer evalDB.Close()

	authorizationHeaders := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		authorizationHeaders <- request.Header.Get("Authorization")
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"decision_result":{"matched_signals":{"domains":["math"]}}}`))
	}))
	defer server.Close()

	scriptPath := filepath.Join(dir, "src", "training", "model_eval", "signal_eval.py")
	if mkdirErr := os.MkdirAll(filepath.Dir(scriptPath), 0o755); mkdirErr != nil {
		t.Fatalf("MkdirAll() error = %v", mkdirErr)
	}
	const script = `
import argparse
import json
import os
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--endpoint")
parser.add_argument("--output")
args, _ = parser.parse_known_args()
token = os.environ["VLLM_SR_EVALUATION_BEARER_TOKEN"]
request = urllib.request.Request(
    args.endpoint,
    data=b"{}",
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    response.read()
with open(args.output, "w", encoding="utf-8") as output:
    json.dump({"dimension": "domain", "total_samples": 1, "correct": 1, "incorrect": 0, "skipped": 0, "accuracy": 1.0}, output)
`
	if writeErr := os.WriteFile(scriptPath, []byte(script), 0o600); writeErr != nil {
		t.Fatalf("WriteFile() error = %v", writeErr)
	}

	task := &models.EvaluationTask{Name: "managed-signal-eval", Config: models.EvaluationConfig{
		Level: models.LevelRouter, Dimensions: []models.EvaluationDimension{models.DimensionDomain},
		Datasets: map[string][]string{string(models.DimensionDomain): {"mmlu-pro-en"}}, Endpoint: server.URL,
	}}
	if createErr := evalDB.CreateTask(task); createErr != nil {
		t.Fatalf("CreateTask() error = %v", createErr)
	}
	runner := NewRunner(RunnerConfig{DB: evalDB, ProjectRoot: dir, ResultsDir: filepath.Join(dir, "results")})
	const delegatedAuthorization = "vsd_managed_runner_fixture"
	authorization, err := NewInferenceAuthorization(delegatedAuthorization)
	if err != nil {
		t.Fatalf("NewInferenceAuthorization() error = %v", err)
	}
	if runErr := runner.RunTask(context.Background(), task.ID, authorization); runErr != nil {
		t.Fatalf("RunTask() error = %v", runErr)
	}
	if header := <-authorizationHeaders; header != "Bearer "+delegatedAuthorization {
		t.Fatalf("Authorization = %q, want delegated bearer", header)
	}

	storedTask, err := evalDB.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask() error = %v", err)
	}
	storedResults, err := evalDB.GetResults(task.ID)
	if err != nil {
		t.Fatalf("GetResults() error = %v", err)
	}
	storedJSON, err := json.Marshal(struct {
		Task    *models.EvaluationTask
		Results []*models.EvaluationResult
	}{storedTask, storedResults})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if strings.Contains(string(storedJSON), delegatedAuthorization) {
		t.Fatal("delegated bearer was persisted in evaluation records")
	}
	err = filepath.Walk(filepath.Join(dir, "results"), func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil || info.IsDir() {
			return walkErr
		}
		contents, readErr := os.ReadFile(path)
		if readErr == nil && strings.Contains(string(contents), delegatedAuthorization) {
			t.Fatalf("delegated bearer was persisted in %s", path)
		}
		return readErr
	})
	if err != nil {
		t.Fatalf("Walk(results) error = %v", err)
	}
}

func TestGetAvailableDatasets_IncludesSignalAndSystemDimensions(t *testing.T) {
	t.Parallel()
	datasets := GetAvailableDatasets()

	if len(datasets[string(models.DimensionDomain)]) == 0 {
		t.Error("expected domain datasets")
	}
	if len(datasets[string(models.DimensionFactCheck)]) == 0 {
		t.Error("expected fact_check datasets")
	}
	if len(datasets[string(models.DimensionUserFeedback)]) == 0 {
		t.Error("expected user_feedback datasets")
	}
	accuracySets := datasets[string(models.DimensionAccuracy)]
	if len(accuracySets) == 0 {
		t.Fatal("expected accuracy (system) datasets")
	}
	found := false
	for _, d := range accuracySets {
		if d.Name == "mmlu-pro" && d.Level == models.LevelMoM {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected mmlu-pro dataset for accuracy (mom level); got %v", accuracySets)
	}
}

func TestRunTaskMarksTaskFailedWhenSystemEvaluationCommandFails(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "eval.db")
	evalDB, err := NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB() error = %v", err)
	}
	defer evalDB.Close()

	task := &models.EvaluationTask{
		Name: "system-eval",
		Config: models.EvaluationConfig{
			Level:         models.LevelMoM,
			Dimensions:    []models.EvaluationDimension{models.DimensionAccuracy},
			Endpoint:      "http://localhost:8801",
			SamplesPerCat: 1,
		},
	}
	createErr := evalDB.CreateTask(task)
	if createErr != nil {
		t.Fatalf("CreateTask() error = %v", createErr)
	}

	runner := NewRunner(RunnerConfig{
		DB:          evalDB,
		ProjectRoot: dir,
		ResultsDir:  filepath.Join(dir, "results"),
		PythonPath:  "python3",
	})

	authorization, authErr := NewInferenceAuthorization("vsd_test_delegated_credential")
	if authErr != nil {
		t.Fatalf("NewInferenceAuthorization() error = %v", authErr)
	}
	err = runner.RunTask(context.Background(), task.ID, authorization)
	if err == nil {
		t.Fatal("RunTask() expected error")
	}

	updatedTask, err := evalDB.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask() error = %v", err)
	}
	if updatedTask == nil {
		t.Fatal("expected task to exist")
	}
	if updatedTask.Status != models.StatusFailed {
		t.Fatalf("task status = %s, want %s", updatedTask.Status, models.StatusFailed)
	}
	if !strings.Contains(updatedTask.ErrorMessage, "system evaluation failed") {
		t.Fatalf("task error message = %q, want substring %q", updatedTask.ErrorMessage, "system evaluation failed")
	}
}
