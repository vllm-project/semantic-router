package handlers

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"
)

func withStatusVersionState(t *testing.T) {
	t.Helper()

	originalVersion := dashboardStatusVersion
	originalSearchRoots := statusVersionSearchRoots
	originalReadBuildInfo := readStatusBuildInfo
	originalVLLMSRVersion, hadVLLMSRVersion := os.LookupEnv("VLLM_SR_VERSION")
	originalDashboardVersion, hadDashboardVersion := os.LookupEnv("DASHBOARD_VERSION")

	dashboardStatusVersion = ""
	statusVersionSearchRoots = func() []string { return nil }
	readStatusBuildInfo = func() (*debug.BuildInfo, bool) { return nil, false }
	_ = os.Unsetenv("VLLM_SR_VERSION")
	_ = os.Unsetenv("DASHBOARD_VERSION")

	t.Cleanup(func() {
		dashboardStatusVersion = originalVersion
		statusVersionSearchRoots = originalSearchRoots
		readStatusBuildInfo = originalReadBuildInfo
		if hadVLLMSRVersion {
			_ = os.Setenv("VLLM_SR_VERSION", originalVLLMSRVersion)
		} else {
			_ = os.Unsetenv("VLLM_SR_VERSION")
		}
		if hadDashboardVersion {
			_ = os.Setenv("DASHBOARD_VERSION", originalDashboardVersion)
		} else {
			_ = os.Unsetenv("DASHBOARD_VERSION")
		}
	})
}

func TestStatusVersionUsesBuildInjectedVersion(t *testing.T) {
	withStatusVersionState(t)

	dashboardStatusVersion = "0.4.0"
	if got := statusVersion(); got != "v0.4.0" {
		t.Fatalf("statusVersion() = %q, want v0.4.0", got)
	}

	dashboardStatusVersion = " v0.5.0-rc.1 "
	if got := statusVersion(); got != "v0.5.0-rc.1" {
		t.Fatalf("statusVersion() = %q, want v0.5.0-rc.1", got)
	}
}

func TestStatusVersionUsesEnvironmentVersion(t *testing.T) {
	withStatusVersionState(t)

	_ = os.Setenv("VLLM_SR_VERSION", "0.6.0")
	if got := statusVersion(); got != "v0.6.0" {
		t.Fatalf("statusVersion() = %q, want v0.6.0", got)
	}

	dashboardStatusVersion = "0.7.0"
	if got := statusVersion(); got != "v0.7.0" {
		t.Fatalf("statusVersion() = %q, want v0.7.0", got)
	}
}

func TestStatusVersionReadsProjectVersionForDevelopment(t *testing.T) {
	withStatusVersionState(t)

	root := t.TempDir()
	projectDir := filepath.Join(root, "src", "vllm-sr")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("create project dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(projectDir, "pyproject.toml"), []byte("[project]\nversion = \"0.8.0\"\n"), 0o644); err != nil {
		t.Fatalf("write pyproject: %v", err)
	}

	statusVersionSearchRoots = func() []string { return []string{filepath.Join(root, "dashboard", "backend")} }
	readStatusBuildInfo = func() (*debug.BuildInfo, bool) {
		return &debug.BuildInfo{Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: "abcdef1234567890"},
		}}, true
	}

	if got := statusVersion(); got != "v0.8.0-dev.abcdef1" {
		t.Fatalf("statusVersion() = %q, want v0.8.0-dev.abcdef1", got)
	}
}

func TestStatusVersionKeepsPrereleaseProjectVersion(t *testing.T) {
	withStatusVersionState(t)

	root := t.TempDir()
	projectDir := filepath.Join(root, "src", "vllm-sr")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("create project dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(projectDir, "pyproject.toml"), []byte("[project]\nversion = \"0.9.0-beta.1\"\n"), 0o644); err != nil {
		t.Fatalf("write pyproject: %v", err)
	}

	statusVersionSearchRoots = func() []string { return []string{root} }

	if got := statusVersion(); got != "v0.9.0-beta.1" {
		t.Fatalf("statusVersion() = %q, want v0.9.0-beta.1", got)
	}
}

func TestStatusVersionMarksDirtyDevelopmentBuild(t *testing.T) {
	withStatusVersionState(t)

	root := t.TempDir()
	projectDir := filepath.Join(root, "src", "vllm-sr")
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		t.Fatalf("create project dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(projectDir, "pyproject.toml"), []byte("[project]\nversion = \"1.0.0\"\n"), 0o644); err != nil {
		t.Fatalf("write pyproject: %v", err)
	}

	statusVersionSearchRoots = func() []string { return []string{root} }
	readStatusBuildInfo = func() (*debug.BuildInfo, bool) {
		return &debug.BuildInfo{Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: "1234567890abcdef"},
			{Key: "vcs.modified", Value: "true"},
		}}, true
	}

	if got := statusVersion(); got != "v1.0.0-dev.1234567.dirty" {
		t.Fatalf("statusVersion() = %q, want v1.0.0-dev.1234567.dirty", got)
	}
}

func TestStatusVersionFallsBackWhenNoVersionSourceExists(t *testing.T) {
	withStatusVersionState(t)

	if got := statusVersion(); got != "unknown" {
		t.Fatalf("statusVersion() = %q, want unknown", got)
	}
}
