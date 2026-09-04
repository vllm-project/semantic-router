package evaluationplane

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCommandProcessFixtureEndToEnd(t *testing.T) {
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the real Python worker")
	}
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation store: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	service, serviceErr := NewService(Options{
		DataDir: root, PythonPath: python, ConfigPath: configPath,
		CodeRevision: testSourceRevision, MaxConcurrent: 1,
	})
	if serviceErr != nil {
		t.Fatalf("NewService: %v", serviceErr)
	}
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Errorf("close evaluation service: %v", err)
		}
	})
	run, createErr := service.CreateRun(context.Background(), CreateRunRequest{
		Name: "real fixture worker", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: append([]TrackID(nil), allTrackIDs...), Mode: ModeReplay, TargetID: "fixture",
		ChangeProfile: "schema_adapter", SampleLimit: 4, Concurrency: 2, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	manifestBefore, beforeErr := os.ReadFile(manifestPath)
	if beforeErr != nil {
		t.Fatalf("read staged manifest: %v", beforeErr)
	}
	// Exercise the production lifecycle. In particular, do not make the
	// server-owned StartedAt equal CreatedAt as that hides worker/report clock
	// authority bugs.
	time.Sleep(time.Millisecond)
	started, startErr := service.StartRun(context.Background(), run.ID)
	if startErr != nil || started.Status != StatusRunning || started.StartedAt == nil || !started.StartedAt.After(run.CreatedAt) {
		t.Fatalf("StartRun=%+v err=%v", started, startErr)
	}
	deadline := time.Now().Add(30 * time.Second)
	for {
		completed, err := service.GetRun(run.ID)
		if err == nil && terminalStatus(completed.Status) {
			if completed.Status != StatusCompleted {
				t.Fatalf("real worker did not complete: %+v", completed)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for real worker: run=%+v err=%v", completed, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	manifestAfter, afterErr := os.ReadFile(manifestPath)
	if afterErr != nil {
		t.Fatalf("read completed manifest: %v", afterErr)
	}
	if !bytes.Equal(manifestBefore, manifestAfter) {
		t.Fatal("Python worker rewrote the server-owned run manifest")
	}
	if _, reportErr := service.ReportJSON(run.ID); reportErr != nil {
		t.Fatalf("strict report validation: %v", reportErr)
	}
	for _, name := range []string{eventsFileName, "events.jsonl", "records.jsonl", reportFileName} {
		if _, err := os.Stat(filepath.Join(root, "runs", run.ID, name)); err != nil {
			t.Fatalf("expected end-to-end bundle file %s: %v", name, err)
		}
	}
}
