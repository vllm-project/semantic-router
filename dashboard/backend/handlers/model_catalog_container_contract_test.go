package handlers

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDashboardContainerUsesOneCatalogCapableRuntime(t *testing.T) {
	repositoryRoot := filepath.Clean(filepath.Join(packageWorkingDirectory(t), "..", "..", ".."))
	legacyDockerfile := filepath.Join(repositoryRoot, "dashboard", "Dockerfile")
	if _, err := os.Stat(legacyDockerfile); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("legacy Dashboard Dockerfile must remain retired, stat error=%v", err)
	}

	canonicalDockerfile := readContainerContractFile(
		t,
		filepath.Join(repositoryRoot, "dashboard", "backend", "Dockerfile"),
	)
	for _, required := range []string{
		"ARG TARGETARCH=amd64",
		"FROM ${IMAGE_REGISTRY}library/python:3.11-slim-bookworm",
		"ENV PYTHONPATH=/app",
		"COPY src/vllm-sr/requirements.txt /app/requirements.txt",
		"COPY src/vllm-sr/pyproject.toml /app/pyproject.toml",
		"COPY src/vllm-sr/cli/ /app/cli/",
		`"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/requirements.txt`,
		"install -d -o nonroot -g root -m 0770 /app/data",
		"find /app/cli -type d -exec chmod 0555 {} +",
		"find /app/cli -type f -exec chmod 0444 {} +",
		"chmod 0555 /app/cli/evaluation/sandbox_worker.py",
	} {
		if !strings.Contains(canonicalDockerfile, required) {
			t.Fatalf("canonical Dashboard Dockerfile omitted runtime contract %q", required)
		}
	}

	openShiftDeploy := readContainerContractFile(
		t,
		filepath.Join(repositoryRoot, "deploy", "openshift", "deploy-to-openshift.sh"),
	)
	if !strings.Contains(openShiftDeploy, `"dockerfilePath":"dashboard/backend/Dockerfile"`) {
		t.Fatal("OpenShift binary build does not use the canonical Dashboard Dockerfile")
	}
	retiredBuildPath := `"dockerfilePath":"dashboard/` + `Dockerfile"`
	if strings.Contains(openShiftDeploy, retiredBuildPath) {
		t.Fatal("OpenShift binary build still references the retired Dashboard Dockerfile")
	}

	entrypoint := readContainerContractFile(
		t,
		filepath.Join(repositoryRoot, "dashboard", "backend", "entrypoint.sh"),
	)
	for _, required := range []string{
		`if [ "$(id -u)" -ne 0 ]; then`,
		"DASHBOARD_RUNTIME_CONFIG_WRITABLE=false",
		"DASHBOARD_RECIPE_STORE_WRITABLE=false",
		`PROVENANCE_FILE_PATH=${CONFIG_FILE_PATH%.*}.provenance.json`,
		`prepare-file "$PROVENANCE_FILE_PATH" "$STATE_GID"`,
		`exec "$@"`,
	} {
		if !strings.Contains(entrypoint, required) {
			t.Fatalf("Dashboard entrypoint omitted arbitrary-UID contract %q", required)
		}
	}

	openShiftDeployment := readContainerContractFile(
		t,
		filepath.Join(repositoryRoot, "deploy", "openshift", "dashboard", "dashboard-deployment.yaml"),
	)
	for _, required := range []string{
		`DASHBOARD_RUNTIME_CONFIG_WRITABLE: "false"`,
		`DASHBOARD_RECIPE_STORE_WRITABLE: "false"`,
		"readOnly: true",
		"emptyDir: {}",
	} {
		if !strings.Contains(openShiftDeployment, required) {
			t.Fatalf("OpenShift Dashboard deployment omitted startup contract %q", required)
		}
	}
}

func packageWorkingDirectory(t *testing.T) string {
	t.Helper()
	workingDirectory, err := os.Getwd()
	if err != nil {
		t.Fatalf("get package working directory: %v", err)
	}
	return workingDirectory
}

func readContainerContractFile(t *testing.T, path string) string {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(contents)
}
