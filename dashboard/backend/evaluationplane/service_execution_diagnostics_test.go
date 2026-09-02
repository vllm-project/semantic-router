package evaluationplane

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

type diagnosticCapture chan string

func (capture diagnosticCapture) Write(payload []byte) (int, error) {
	capture <- string(payload)
	return len(payload), nil
}

func TestBuildTerminalRunReturnsSynthesizedFailureCause(t *testing.T) {
	run := Run{
		ID:       newTestClientRequestID(),
		Status:   StatusRunning,
		Progress: RunProgress{Total: 1},
	}
	terminal, cause := (&Service{}).buildTerminalRun(run, nil)
	if terminal.Status != StatusFailed || cause == nil ||
		!strings.Contains(cause.Error(), "not in the sealing phase") {
		t.Fatalf("terminal run=%+v cause=%v, want protected synthesized failure", terminal, cause)
	}
}

func TestExecutionDiagnosticsStayInTheServiceOwnedSink(t *testing.T) {
	release := make(chan struct{})
	close(release)
	privateFailure := "provider-private-routing-diagnostic"
	service, _ := newTestService(t, &controlledProcess{
		release: release,
		err:     errors.New(privateFailure),
	}, 1)
	defer func() { _ = service.Close() }()
	diagnostics := make(diagnosticCapture, 1)
	service.diagnosticLogger.SetOutput(diagnostics)

	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	failed := waitForRunStatus(t, service, run.ID, StatusFailed)
	if !strings.Contains(failed.Error, "protected server diagnostics") ||
		strings.Contains(failed.Error, privateFailure) {
		t.Fatalf("public terminal error leaked protected detail: %q", failed.Error)
	}
	select {
	case diagnostic := <-diagnostics:
		if !strings.Contains(diagnostic, run.ID) || !strings.Contains(diagnostic, privateFailure) {
			t.Fatalf("protected diagnostic sink omitted execution context: %q", diagnostic)
		}
	case <-time.After(time.Second):
		t.Fatal("protected diagnostic sink did not receive the execution failure")
	}
}
