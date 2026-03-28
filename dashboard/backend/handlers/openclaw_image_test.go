package handlers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFakeOpenClawRuntime(t *testing.T) (runtimePath string, logPath string) {
	t.Helper()

	tempDir := t.TempDir()
	logPath = filepath.Join(tempDir, "runtime.log")
	runtimePath = filepath.Join(tempDir, "fake-runtime.sh")
	script := `#!/bin/sh
echo "$@" >> "$TEST_RUNTIME_LOG"
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  if [ "$TEST_IMAGE_INSPECT_OK" = "1" ]; then
    exit 0
  fi
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "ls" ]; then
  if [ -n "$TEST_IMAGE_LS_OUTPUT" ]; then
    printf "%s\n" "$TEST_IMAGE_LS_OUTPUT"
  fi
  exit 0
fi
if [ "$1" = "pull" ]; then
  if [ "$TEST_PULL_OK" = "1" ]; then
    echo "pulled $2"
    exit 0
  fi
  if [ -n "$TEST_PULL_ERROR" ]; then
    echo "$TEST_PULL_ERROR" >&2
  fi
  exit 1
fi
exit 0
`
	if err := os.WriteFile(runtimePath, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake runtime: %v", err)
	}
	return runtimePath, logPath
}

func readFakeRuntimeLog(t *testing.T, logPath string) string {
	t.Helper()

	data, err := os.ReadFile(logPath)
	if err != nil {
		if os.IsNotExist(err) {
			return ""
		}
		t.Fatalf("failed to read fake runtime log: %v", err)
	}
	return string(data)
}

func TestEnsureImageAvailable_AlwaysPullsRemoteImage(t *testing.T) {
	runtimePath, logPath := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("TEST_RUNTIME_LOG", logPath)
	t.Setenv("TEST_IMAGE_INSPECT_OK", "1")
	t.Setenv("TEST_PULL_OK", "1")

	h := NewOpenClawHandler(t.TempDir(), false)
	image := "ghcr.io/openclaw/openclaw:latest"
	if err := h.ensureImageAvailable(image); err != nil {
		t.Fatalf("ensureImageAvailable failed: %v", err)
	}

	logOutput := readFakeRuntimeLog(t, logPath)
	if !strings.Contains(logOutput, "pull "+image) {
		t.Fatalf("expected runtime to pull image %q, got log:\n%s", image, logOutput)
	}
}

func TestResolveBaseImage_KeepsDefaultInsteadOfAutoSelectingLocalFallback(t *testing.T) {
	runtimePath, logPath := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("TEST_RUNTIME_LOG", logPath)
	t.Setenv("TEST_IMAGE_LS_OUTPUT", "local/openclaw:dev")

	h := NewOpenClawHandler(t.TempDir(), false)
	if got := h.resolveBaseImage(""); got != "ghcr.io/openclaw/openclaw:latest" {
		t.Fatalf("expected default image, got %q", got)
	}

	logOutput := readFakeRuntimeLog(t, logPath)
	if strings.Contains(logOutput, "image ls") {
		t.Fatalf("resolveBaseImage should not probe local image inventory anymore, got log:\n%s", logOutput)
	}
}

func TestEnsureImageAvailable_LocalTagStillRequiresLocalImage(t *testing.T) {
	runtimePath, logPath := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("TEST_RUNTIME_LOG", logPath)

	h := NewOpenClawHandler(t.TempDir(), false)
	err := h.ensureImageAvailable("demo/openclaw:local")
	if err == nil {
		t.Fatalf("expected local-only image check to fail")
	}
	if !strings.Contains(err.Error(), "missing locally and cannot be auto-pulled") {
		t.Fatalf("unexpected error: %v", err)
	}

	logOutput := readFakeRuntimeLog(t, logPath)
	if strings.Contains(logOutput, "pull demo/openclaw:local") {
		t.Fatalf("local-only image should not be pulled, got log:\n%s", logOutput)
	}
}

func TestDetectContainerRuntimeUsesExplicitRuntimePath(t *testing.T) {
	runtimePath, _ := writeFakeOpenClawRuntime(t)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("CONTAINER_RUNTIME", "")

	got, err := detectContainerRuntime()
	if err != nil {
		t.Fatalf("detectContainerRuntime failed: %v", err)
	}
	if got != runtimePath {
		t.Fatalf("runtime path = %q, want %q", got, runtimePath)
	}
}

func TestDetectContainerRuntimeRejectsPodmanOverride(t *testing.T) {
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", "podman")
	t.Setenv("CONTAINER_RUNTIME", "")

	_, err := detectContainerRuntime()
	if err == nil {
		t.Fatalf("expected podman override to be rejected")
	}
	if !strings.Contains(err.Error(), "podman is not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDetectContainerRuntimeRejectsPodmanContainerRuntimeEnv(t *testing.T) {
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", "")
	t.Setenv("CONTAINER_RUNTIME", "podman")

	_, err := detectContainerRuntime()
	if err == nil {
		t.Fatalf("expected podman CONTAINER_RUNTIME to be rejected")
	}
	if !strings.Contains(err.Error(), "podman is not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}
