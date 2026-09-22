package framework

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
)

func TestCreateWorkspaceModelsValuesFile(t *testing.T) {
	valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
	if err != nil {
		t.Fatalf("createWorkspaceModelsValuesFile returned error: %v", err)
	}
	defer cleanup()

	content, err := os.ReadFile(valuesFile)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}

	values := string(content)
	if !strings.Contains(values, "persistence:\n  enabled: false") {
		t.Fatalf("expected workspace models overlay to disable persistence, got:\n%s", values)
	}
	if !strings.Contains(values, "mountPath: /app/models") {
		t.Fatalf("expected workspace models overlay to mount /app/models, got:\n%s", values)
	}
	if !strings.Contains(values, "path: "+cluster.WorkspaceModelsNodeMountPath) {
		t.Fatalf("expected workspace models overlay to use %q, got:\n%s", cluster.WorkspaceModelsNodeMountPath, values)
	}
}

func TestCreateWorkspaceModelsValuesFileWithoutHFEndpoint(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "")

	valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
	if err != nil {
		t.Fatalf("createWorkspaceModelsValuesFile returned error: %v", err)
	}
	defer cleanup()

	content, err := os.ReadFile(valuesFile)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}

	values := string(content)
	if strings.Contains(values, "HF_ENDPOINT") {
		t.Fatalf("expected workspace models overlay to omit HF_ENDPOINT when env is empty, got:\n%s", values)
	}
}

func TestCreateWorkspaceModelsValuesFileWithHFEndpoint(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "https://hf-mirror.com")

	valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
	if err != nil {
		t.Fatalf("createWorkspaceModelsValuesFile returned error: %v", err)
	}
	defer cleanup()

	content, err := os.ReadFile(valuesFile)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}

	values := string(content)
	if !strings.Contains(values, "extraEnv:") {
		t.Fatalf("expected workspace models overlay to contain extraEnv, got:\n%s", values)
	}
	if !strings.Contains(values, "- name: HF_ENDPOINT") {
		t.Fatalf("expected extraEnv to contain HF_ENDPOINT, got:\n%s", values)
	}
	if !strings.Contains(values, `value: "https://hf-mirror.com"`) {
		t.Fatalf("expected extraEnv to contain mirror value, got:\n%s", values)
	}
}

func TestResolveWorkspaceModelsDirPermissions(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(origDir) }()

	modelsDir, err := resolveWorkspaceModelsDir()
	if err != nil {
		t.Fatalf("resolveWorkspaceModelsDir returned error: %v", err)
	}

	info, err := os.Stat(modelsDir)
	if err != nil {
		t.Fatalf("stat %s returned error: %v", modelsDir, err)
	}

	if perm := info.Mode().Perm(); perm&0o777 != 0o777 {
		t.Fatalf("expected modelsDir perm to be 0777, got %04o", perm)
	}
}

func TestWorkspaceModelsValuesFileHelmDeduplication(t *testing.T) {
	if _, err := exec.LookPath("helm"); err != nil {
		t.Skip("helm executable not found in PATH")
	}

	valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
	if err != nil {
		t.Fatalf("createWorkspaceModelsValuesFile returned error: %v", err)
	}
	defer cleanup()

	// Locate chart directory relative to e2e/pkg/framework
	chartDir := filepath.Join("..", "..", "..", "deploy", "helm", "semantic-router")
	if _, err := os.Stat(chartDir); err != nil {
		t.Fatalf("chart directory not found at %s: %v", chartDir, err)
	}

	tmpChartDir := filepath.Join(t.TempDir(), "semantic-router")
	if err := copyDirWithoutDependencies(chartDir, tmpChartDir); err != nil {
		t.Fatalf("prepare temporary chart: %v", err)
	}

	cmd := exec.Command("helm", "template", "dedup-test", tmpChartDir, "-f", valuesFile, "-s", "templates/deployment.yaml")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("helm template failed: %v\nOutput:\n%s", err, string(out))
	}

	manifest := string(out)
	if strings.Contains(manifest, "name: models-volume") {
		t.Fatalf("expected default models-volume to be omitted when workspace models are mounted, got manifest:\n%s", manifest)
	}
	if !strings.Contains(manifest, "name: workspace-models") {
		t.Fatalf("expected workspace-models mount to be present, got manifest:\n%s", manifest)
	}
	if count := strings.Count(manifest, "mountPath: /app/models"); count != 1 {
		t.Fatalf("expected exactly 1 mountPath: /app/models, got %d in manifest:\n%s", count, manifest)
	}
}

func copyDirWithoutDependencies(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			if info.Name() == "charts" {
				return filepath.SkipDir
			}
			return os.MkdirAll(target, info.Mode())
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if rel == "Chart.yaml" {
			lines := strings.Split(string(data), "\n")
			var filtered []string
			inDep := false
			for _, line := range lines {
				if strings.HasPrefix(line, "dependencies:") {
					inDep = true
					continue
				}
				if inDep {
					if strings.HasPrefix(line, "  ") || strings.HasPrefix(line, "\t") {
						continue
					}
					inDep = false
				}
				filtered = append(filtered, line)
			}
			data = []byte(strings.Join(filtered, "\n"))
		}
		return os.WriteFile(target, data, info.Mode())
	})
}

func TestSetupProfileRegistersTeardownCleanupByDefault(t *testing.T) {
	runner := &Runner{
		opts:    &TestOptions{},
		profile: &stubProfile{},
	}
	state := &runState{}

	if err := runner.setupProfile(context.Background(), state); err != nil {
		t.Fatalf("setupProfile returned error: %v", err)
	}

	if len(state.cleanup) != 1 {
		t.Fatalf("expected teardown cleanup to be registered, got %d cleanups", len(state.cleanup))
	}
}

func TestSetupProfileSkipsTeardownCleanupWhenKeepingCluster(t *testing.T) {
	runner := &Runner{
		opts:    &TestOptions{KeepCluster: true},
		profile: &stubProfile{},
	}
	state := &runState{}

	if err := runner.setupProfile(context.Background(), state); err != nil {
		t.Fatalf("setupProfile returned error: %v", err)
	}

	if len(state.cleanup) != 0 {
		t.Fatalf("expected no teardown cleanup when keeping cluster, got %d cleanups", len(state.cleanup))
	}
}

type stubProfile struct{}

func (p *stubProfile) Name() string { return "stub" }

func (p *stubProfile) Setup(context.Context, *SetupOptions) error { return nil }

func (p *stubProfile) Teardown(context.Context, *TeardownOptions) error { return nil }

func (p *stubProfile) GetTestCases() []string { return nil }

func (p *stubProfile) GetServiceConfig() ServiceConfig { return ServiceConfig{} }

func TestSkipMissingLocalImageDeployment(t *testing.T) {
	notFound := apierrors.NewNotFound(schema.GroupResource{Resource: "deployments"}, "mock-vllm")
	if !skipMissingLocalImageDeployment(notFound) {
		t.Fatal("expected a missing deployment to be skipped")
	}
	if skipMissingLocalImageDeployment(fmt.Errorf("connection refused")) {
		t.Fatal("expected other lookup errors to fail the restart")
	}
}
