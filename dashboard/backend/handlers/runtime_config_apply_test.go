package handlers

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestValidateManagedContainerExecArgsAllowsVirtualenvPython(t *testing.T) {
	err := validateManagedContainerExecArgs([]string{
		"/opt/vllm-sr-dashboard-venv/bin/python3",
		"-c",
		"print('ok')",
	})
	if err != nil {
		t.Fatalf("expected virtualenv python command to be allowed, got %v", err)
	}
}

func TestIsPythonCommandDetectsAbsolutePythonPath(t *testing.T) {
	if !isPythonCommand("/opt/vllm-sr-dashboard-venv/bin/python3") {
		t.Fatal("expected absolute virtualenv python path to be recognized")
	}
	if isPythonCommand("/bin/sh") {
		t.Fatal("did not expect non-python binary to be recognized")
	}
}

func TestApplyWrittenConfigRemovesCandidateWhenPreviousConfigWasAbsent(t *testing.T) {
	useUnknownManagedDockerCLI(t)
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	previous, err := captureConfigFileSnapshot(configPath)
	if err != nil {
		t.Fatalf("capture absent config: %v", err)
	}
	if previous.existed {
		t.Fatal("missing config snapshot unexpectedly reports an existing file")
	}

	candidatePath := createValidTestConfig(t, tempDir)
	if err := applyWrittenConfig(candidatePath, tempDir, previous, true); err == nil {
		t.Fatal("expected runtime apply failure")
	}
	if _, err := os.Stat(configPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("candidate config still exists after rollback: %v", err)
	}
}

func TestApplyWrittenConfigRestoresExistingZeroByteConfig(t *testing.T) {
	useUnknownManagedDockerCLI(t)
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, nil, 0o644); err != nil {
		t.Fatalf("write empty config: %v", err)
	}
	previous, captureErr := captureConfigFileSnapshot(configPath)
	if captureErr != nil {
		t.Fatalf("capture empty config: %v", captureErr)
	}
	if !previous.existed || len(previous.data) != 0 {
		t.Fatalf("empty config snapshot = existed:%v bytes:%d", previous.existed, len(previous.data))
	}

	createValidTestConfig(t, tempDir)
	if err := applyWrittenConfig(configPath, tempDir, previous, true); err == nil {
		t.Fatal("expected runtime apply failure")
	}
	restored, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read restored empty config: %v", err)
	}
	if len(restored) != 0 {
		t.Fatalf("restored config has %d bytes, want zero", len(restored))
	}
}
