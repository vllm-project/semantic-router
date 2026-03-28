package evaluation

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func TestRecoverRunningTasks(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "eval.db")
	evalDB, err := NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB() error = %v", err)
	}
	defer evalDB.Close()

	task := &models.EvaluationTask{
		Name:   "recover-test",
		Config: models.EvaluationConfig{Level: models.LevelRouter, Dimensions: []models.EvaluationDimension{models.DimensionDomain}},
	}
	if err = evalDB.CreateTask(task); err != nil {
		t.Fatalf("CreateTask() error = %v", err)
	}

	if err = evalDB.UpdateTaskStatus(task.ID, models.StatusRunning, ""); err != nil {
		t.Fatalf("UpdateTaskStatus(running) error = %v", err)
	}

	msg := "Dashboard restarted; task interrupted"
	if err = evalDB.RecoverRunningTasks(msg); err != nil {
		t.Fatalf("RecoverRunningTasks() error = %v", err)
	}

	got, err := evalDB.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask() error = %v", err)
	}
	if got.Status != models.StatusFailed {
		t.Errorf("status = %q, want %q", got.Status, models.StatusFailed)
	}
	if got.ErrorMessage != msg {
		t.Errorf("error_message = %q, want %q", got.ErrorMessage, msg)
	}
}

func TestCreateTaskAndGetTask(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "eval.db")
	evalDB, err := NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB() error = %v", err)
	}
	defer evalDB.Close()

	task := &models.EvaluationTask{
		Name:        "signal-eval",
		Description: "Test signal evaluation",
		Config: models.EvaluationConfig{
			Level:      models.LevelRouter,
			Dimensions: []models.EvaluationDimension{models.DimensionDomain, models.DimensionFactCheck},
			Endpoint:   "http://localhost:8080/api/v1/eval",
			MaxSamples: 50,
		},
	}
	if err = evalDB.CreateTask(task); err != nil {
		t.Fatalf("CreateTask() error = %v", err)
	}
	if task.ID == "" {
		t.Error("CreateTask() did not set task ID")
	}
	if task.Status != models.StatusPending {
		t.Errorf("status = %q, want pending", task.Status)
	}

	got, err := evalDB.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask() error = %v", err)
	}
	if got.Name != task.Name || got.Status != models.StatusPending {
		t.Errorf("GetTask() name=%q status=%q, want name=%q status=pending", got.Name, got.Status, task.Name)
	}
	if len(got.Config.Dimensions) != 2 {
		t.Errorf("len(Config.Dimensions) = %d, want 2", len(got.Config.Dimensions))
	}
}

func TestCreateTask_SystemLevelAccuracy(t *testing.T) {
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
			Level:      models.LevelMoM,
			Dimensions: []models.EvaluationDimension{models.DimensionAccuracy},
			Endpoint:   "http://localhost:8801",
			MaxSamples: 20,
		},
	}
	if err = evalDB.CreateTask(task); err != nil {
		t.Fatalf("CreateTask() error = %v", err)
	}
	if task.ID == "" {
		t.Error("CreateTask() did not set task ID")
	}

	got, err := evalDB.GetTask(task.ID)
	if err != nil {
		t.Fatalf("GetTask() error = %v", err)
	}
	if got.Config.Level != models.LevelMoM || len(got.Config.Dimensions) != 1 || got.Config.Dimensions[0] != models.DimensionAccuracy {
		t.Errorf("GetTask() level=%q dimensions=%v, want mom and [accuracy]", got.Config.Level, got.Config.Dimensions)
	}
}
