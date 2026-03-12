package handlers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultOpenClawClawHubSkills_DefaultsToSonosCLI(t *testing.T) {
	original, hadOriginal := os.LookupEnv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS")
	if err := os.Unsetenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS"); err != nil {
		t.Fatalf("failed to clear env: %v", err)
	}
	defer func() {
		if hadOriginal {
			_ = os.Setenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS", original)
			return
		}
		_ = os.Unsetenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS")
	}()

	gotDefault := defaultOpenClawClawHubSkills()
	if len(gotDefault) != 1 || gotDefault[0] != "sonoscli" {
		t.Fatalf("expected default skill list [sonoscli], got %v", gotDefault)
	}

	t.Setenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS", "")
	if got := defaultOpenClawClawHubSkills(); got != nil {
		t.Fatalf("explicit empty env should disable defaults, got %v", got)
	}

	t.Setenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS", "sonoscli, sonoscli other-skill")
	got := defaultOpenClawClawHubSkills()
	if len(got) != 2 || got[0] != "sonoscli" || got[1] != "other-skill" {
		t.Fatalf("unexpected parsed default skills: %v", got)
	}
}

func TestNormalizeOpenClawClawHubSkills_DisabledSentinels(t *testing.T) {
	for _, raw := range []string{"none", "off", "disabled", "false", "   "} {
		if got := normalizeOpenClawClawHubSkills(raw); got != nil {
			t.Fatalf("expected %q to disable defaults, got %v", raw, got)
		}
	}
}

func TestInstallDefaultClawHubSkills_RunsClawHubInstallIntoWorkspace(t *testing.T) {
	runtimePath, logPath := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("TEST_RUNTIME_LOG", logPath)
	t.Setenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS", "sonoscli")

	workspaceDir := filepath.Join(t.TempDir(), "workspace")
	h := NewOpenClawHandler(t.TempDir(), false)
	if err := h.installDefaultClawHubSkills(workspaceDir, "ghcr.io/openclaw/openclaw:latest"); err != nil {
		t.Fatalf("installDefaultClawHubSkills failed: %v", err)
	}

	logOutput := readFakeRuntimeLog(t, logPath)
	expectedParts := []string{
		"run --rm --entrypoint npx",
		"ghcr.io/openclaw/openclaw:latest",
		"--yes clawhub@latest",
		"--workdir /workspace --no-input",
		"install sonoscli --force",
	}
	for _, part := range expectedParts {
		if !strings.Contains(logOutput, part) {
			t.Fatalf("expected runtime log to contain %q, got:\n%s", part, logOutput)
		}
	}
}

func TestInstallDefaultClawHubSkills_DisabledSkipsRuntimeCall(t *testing.T) {
	runtimePath, logPath := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("TEST_RUNTIME_LOG", logPath)
	t.Setenv("OPENCLAW_DEFAULT_CLAWHUB_SKILLS", "disabled")

	workspaceDir := filepath.Join(t.TempDir(), "workspace")
	h := NewOpenClawHandler(t.TempDir(), false)
	if err := h.installDefaultClawHubSkills(workspaceDir, "ghcr.io/openclaw/openclaw:latest"); err != nil {
		t.Fatalf("installDefaultClawHubSkills should no-op when disabled: %v", err)
	}

	if logOutput := readFakeRuntimeLog(t, logPath); strings.TrimSpace(logOutput) != "" {
		t.Fatalf("expected no runtime calls when defaults are disabled, got:\n%s", logOutput)
	}
}
