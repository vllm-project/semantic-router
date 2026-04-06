package evaluation

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingartifacts"
)

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

	err = runner.RunTask(context.Background(), task.ID)
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

func TestModelEvalScriptPathUsesSharedArtifactContract(t *testing.T) {
	t.Parallel()

	runner := NewRunner(RunnerConfig{ProjectRoot: "/repo"})
	want, err := trainingartifacts.CurrentContract().ModelEvalScriptPath(
		"/repo",
		trainingartifacts.ModelEvalSignalEvalScript,
	)
	if err != nil {
		t.Fatalf("ModelEvalScriptPath(): %v", err)
	}
	if got := runner.modelEvalScriptPath(trainingartifacts.ModelEvalSignalEvalScript); got != want {
		t.Fatalf("modelEvalScriptPath() = %q, want %q", got, want)
	}
}
