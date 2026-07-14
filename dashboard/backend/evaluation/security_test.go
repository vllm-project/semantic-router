package evaluation

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func TestNewDBUsesPrivateModes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits are required")
	}
	dir := filepath.Join(t.TempDir(), "evaluation-private")
	dbPath := filepath.Join(dir, "eval.db")
	db, err := NewDB(dbPath)
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	assertEvaluationMode(t, dir, evaluationPrivateDirMode)
	assertEvaluationMode(t, dbPath, evaluationPrivateFileMode)
	for _, suffix := range []string{"-wal", "-shm"} {
		if _, err := os.Lstat(dbPath + suffix); err == nil {
			assertEvaluationMode(t, dbPath+suffix, evaluationPrivateFileMode)
		}
	}
}

func TestNewDBRejectsDatabaseAndSidecarSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic-link behavior differs on Windows")
	}
	for _, targetKind := range []string{"database", "wal-sidecar"} {
		t.Run(targetKind, func(t *testing.T) {
			dir := t.TempDir()
			dbPath := filepath.Join(dir, "eval.db")
			external := filepath.Join(t.TempDir(), "external")
			if err := os.WriteFile(external, []byte("sentinel"), 0o600); err != nil {
				t.Fatal(err)
			}
			linkPath := dbPath
			if targetKind == "wal-sidecar" {
				linkPath += "-wal"
			}
			if err := os.Symlink(external, linkPath); err != nil {
				t.Fatal(err)
			}
			if db, err := NewDB(dbPath); err == nil {
				_ = db.Close()
				t.Fatal("NewDB unexpectedly accepted symbolic link")
			}
			got, err := os.ReadFile(external)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != "sentinel" {
				t.Fatalf("database initialization followed symlink: %q", got)
			}
		})
	}
}

func TestRunnerPythonEnvironmentDoesNotInheritSecretsOrProxies(t *testing.T) {
	t.Setenv("DASHBOARD_AUTH_PASSWORD", "sentinel-dashboard-secret")
	t.Setenv("OPENAI_API_KEY", "sentinel-provider-secret")
	t.Setenv("HTTPS_PROXY", "http://sentinel-proxy.invalid")
	t.Setenv("PATH", os.Getenv("PATH"))
	runner := NewRunner(RunnerConfig{ProjectRoot: t.TempDir(), ResultsDir: filepath.Join(t.TempDir(), "results")})
	env := strings.Join(runner.pythonEnv(), "\n")
	for _, forbidden := range []string{"DASHBOARD_AUTH_PASSWORD=", "OPENAI_API_KEY=", "HTTPS_PROXY=", "sentinel-"} {
		if strings.Contains(env, forbidden) {
			t.Fatalf("subprocess environment inherited forbidden value %q: %s", forbidden, env)
		}
	}
	if !strings.Contains(env, "PATH=") || !strings.Contains(env, "PYTHONPATH=") {
		t.Fatalf("subprocess environment omitted required runtime paths: %s", env)
	}
}

func TestRunCommandBoundsStdoutAndRedactsStderr(t *testing.T) {
	runner := &Runner{}
	success := exec.Command(os.Args[0], "-test.run=TestEvaluationCommandHelperProcess") //nolint:gosec // os.Args[0] is the trusted current test binary.
	success.Env = append(os.Environ(), "GO_WANT_EVALUATION_HELPER=success")
	output, err := runner.runCommandWithProgress(context.Background(), success, "task", "dimension")
	if err != nil {
		t.Fatalf("bounded success command failed: %v", err)
	}
	if len(output) != evaluationMaxCapturedStdoutBytes {
		t.Fatalf("captured stdout bytes = %d, want %d", len(output), evaluationMaxCapturedStdoutBytes)
	}

	failure := exec.Command(os.Args[0], "-test.run=TestEvaluationCommandHelperProcess") //nolint:gosec // os.Args[0] is the trusted current test binary.
	failure.Env = append(os.Environ(), "GO_WANT_EVALUATION_HELPER=failure")
	_, err = runner.runCommandWithProgress(context.Background(), failure, "task", "dimension")
	if err == nil {
		t.Fatal("failing helper unexpectedly succeeded")
	}
	if strings.Contains(err.Error(), "sentinel-stderr-secret") {
		t.Fatalf("command error leaked stderr: %v", err)
	}
}

func TestEvaluationCommandHelperProcess(t *testing.T) {
	switch os.Getenv("GO_WANT_EVALUATION_HELPER") {
	case "success":
		_, _ = io.WriteString(os.Stdout, strings.Repeat("x", evaluationMaxCapturedStdoutBytes+64<<10))
	case "failure":
		_, _ = io.WriteString(os.Stderr, "sentinel-stderr-secret /private/server/path\n")
		os.Exit(17)
	}
}

func TestRunTaskPersistsGenericFailureAndPrivateResults(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses a POSIX executable script and permission bits")
	}
	dir := t.TempDir()
	db, err := NewDB(filepath.Join(dir, "eval.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	task := &models.EvaluationTask{
		Name: "generic-failure",
		Config: models.EvaluationConfig{
			Level:         models.LevelMoM,
			Dimensions:    []models.EvaluationDimension{models.DimensionAccuracy},
			Endpoint:      "http://localhost:8801",
			SamplesPerCat: 1,
		},
	}
	if createErr := db.CreateTask(task); createErr != nil {
		t.Fatal(createErr)
	}
	fakePython := filepath.Join(dir, "fake-python")
	if writeErr := os.WriteFile(fakePython, []byte("#!/bin/sh\necho 'sentinel-provider-body /private/path' >&2\nexit 9\n"), 0o700); writeErr != nil {
		t.Fatal(writeErr)
	}
	resultsDir := filepath.Join(dir, "results")
	runner := NewRunner(RunnerConfig{DB: db, ProjectRoot: dir, ResultsDir: resultsDir, PythonPath: fakePython})
	if runErr := runner.RunTask(context.Background(), task.ID); runErr == nil {
		t.Fatal("RunTask unexpectedly succeeded")
	}
	updated, err := db.GetTask(task.ID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.ErrorMessage != "system evaluation failed" {
		t.Fatalf("persisted error = %q, want generic stage error", updated.ErrorMessage)
	}
	if strings.Contains(updated.ErrorMessage, "sentinel") || strings.Contains(updated.ErrorMessage, "/private/") {
		t.Fatalf("persisted error leaked subprocess content: %q", updated.ErrorMessage)
	}
	assertEvaluationMode(t, resultsDir, evaluationPrivateDirMode)
	assertEvaluationMode(t, filepath.Join(resultsDir, task.ID), evaluationPrivateDirMode)
}

func TestRunTaskRejectsPersistedDatasetTraversalBeforeCreatingArtifacts(t *testing.T) {
	dir := t.TempDir()
	db, err := NewDB(filepath.Join(dir, "eval.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	task := &models.EvaluationTask{
		Name: "legacy-unsafe-dataset",
		Config: models.EvaluationConfig{
			Level:      models.LevelRouter,
			Dimensions: []models.EvaluationDimension{models.DimensionDomain},
			Datasets:   map[string][]string{"domain": {"../../sentinel"}},
			Endpoint:   "http://localhost:8080/api/v1/eval",
		},
	}
	if createErr := db.CreateTask(task); createErr != nil {
		t.Fatal(createErr)
	}
	resultsDir := filepath.Join(dir, "results")
	runner := NewRunner(RunnerConfig{DB: db, ProjectRoot: dir, ResultsDir: resultsDir})
	if runErr := runner.RunTask(context.Background(), task.ID); runErr == nil {
		t.Fatal("unsafe persisted dataset unexpectedly ran")
	}
	if _, statErr := os.Lstat(filepath.Join(dir, "sentinel.json")); !os.IsNotExist(statErr) {
		t.Fatalf("dataset traversal created an artifact outside task directory: %v", statErr)
	}
	updated, err := db.GetTask(task.ID)
	if err != nil {
		t.Fatal(err)
	}
	if updated.ErrorMessage != "Evaluation configuration is invalid" {
		t.Fatalf("persisted error = %q, want generic configuration error", updated.ErrorMessage)
	}
}

func assertEvaluationMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if got := info.Mode().Perm(); got != want {
		t.Fatalf("mode(%s) = %o, want %o", path, got, want)
	}
}
