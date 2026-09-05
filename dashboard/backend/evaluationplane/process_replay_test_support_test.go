package evaluationplane

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func runCommandProcessReplay(t *testing.T, request CreateRunRequest) (Report, []byte) {
	t.Helper()
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the real Python worker")
	}
	pythonRoot, pathErr := filepath.Abs("../../../src/vllm-sr")
	if pathErr != nil {
		t.Fatal(pathErr)
	}
	t.Setenv("PYTHONPATH", pythonRoot)
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
	var workerDiagnostics bytes.Buffer
	service.process.(*CommandProcess).diagnosticSink = &workerDiagnostics
	processCapture := &capturedProcess{Process: service.process}
	service.process = processCapture
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Errorf("close evaluation service: %v", err)
		}
	})
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), request)
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
	started, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID)
	if startErr != nil || started.Status != StatusRunning || started.StartedAt == nil || !started.StartedAt.After(run.CreatedAt) {
		t.Fatalf("StartRun=%+v err=%v", started, startErr)
	}
	waitForCompletedFixtureRun(t, service, run.ID, processCapture, &workerDiagnostics)
	manifestAfter, afterErr := os.ReadFile(manifestPath)
	if afterErr != nil {
		t.Fatalf("read completed manifest: %v", afterErr)
	}
	if !bytes.Equal(manifestBefore, manifestAfter) {
		t.Fatal("Python worker rewrote the server-owned run manifest")
	}
	reportBytes, reportErr := service.ReportJSONAs(SystemActor(), run.ID)
	if reportErr != nil {
		t.Fatalf("strict report validation: %v", reportErr)
	}
	report, decodeErr := decodeReportStrict(run.ID, reportBytes)
	if decodeErr != nil {
		t.Fatalf("decode server-sealed report: %v", decodeErr)
	}
	if report.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("server-sealed report attestation_revision=%q, want %q", report.AttestationRevision, ServerAttestationRevision)
	}
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatalf("read server-owned report anchor: %v", anchorErr)
	}
	if anchor.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("server-owned anchor attestation_revision=%q, want %q", anchor.AttestationRevision, ServerAttestationRevision)
	}
	for _, name := range []string{eventsFileName, "records.jsonl", reportFileName} {
		if _, err := os.Stat(filepath.Join(root, "runs", run.ID, name)); err != nil {
			t.Fatalf("expected end-to-end bundle file %s: %v", name, err)
		}
	}
	records, err := os.ReadFile(filepath.Join(root, "runs", run.ID, "records.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	return report, records
}
