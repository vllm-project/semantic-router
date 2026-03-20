package framework

import (
	"context"
	"os"
	"strings"
	"testing"

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
