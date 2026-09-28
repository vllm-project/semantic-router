package framework

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
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

func TestCreateWorkspaceModelsValuesFileHFEndpoint(t *testing.T) {
	for _, tc := range []struct {
		name     string
		endpoint string
	}{
		{name: "unset"},
		{name: "mirror", endpoint: "https://mirror.example/hf?region=a:b"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("HF_ENDPOINT", tc.endpoint)
			valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
			if err != nil {
				t.Fatal(err)
			}
			defer cleanup()
			content, err := os.ReadFile(valuesFile)
			if err != nil {
				t.Fatal(err)
			}
			var values struct {
				ExtraEnv []struct {
					Name  string `yaml:"name"`
					Value string `yaml:"value"`
				} `yaml:"extraEnv"`
			}
			if err := yaml.Unmarshal(content, &values); err != nil {
				t.Fatalf("invalid workspace values YAML: %v", err)
			}
			if tc.endpoint == "" {
				if len(values.ExtraEnv) != 0 {
					t.Fatalf("unexpected extraEnv when HF_ENDPOINT is unset: %+v", values.ExtraEnv)
				}
				return
			}
			if len(values.ExtraEnv) != 1 || values.ExtraEnv[0].Name != "HF_ENDPOINT" || values.ExtraEnv[0].Value != tc.endpoint {
				t.Fatalf("HF_ENDPOINT was not forwarded exactly: %+v", values.ExtraEnv)
			}
		})
	}
}

func TestEnsureWorkspaceModelsDirSetsPermissiveMode(t *testing.T) {
	modelsDir := filepath.Join(t.TempDir(), "models")
	if err := os.Mkdir(modelsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := ensureWorkspaceModelsDir(modelsDir); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(modelsDir)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o777 {
		t.Fatalf("workspace models permissions = %04o, want 0777", got)
	}
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
	notFound := apierrors.NewNotFound(schema.GroupResource{Resource: "deployments"}, "provider-mocker")
	if !skipMissingLocalImageDeployment(notFound) {
		t.Fatal("expected a missing deployment to be skipped")
	}
	if skipMissingLocalImageDeployment(fmt.Errorf("connection refused")) {
		t.Fatal("expected other lookup errors to fail the restart")
	}
}
