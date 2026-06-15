package cluster

import (
	"os"
	"strings"
	"testing"
)

func TestCreateClusterConfigWithoutWorkspaceModelsMount(t *testing.T) {
	cluster := NewKindCluster("unit-test", false)

	configFile, err := cluster.createClusterConfig()
	if err != nil {
		t.Fatalf("createClusterConfig returned error: %v", err)
	}
	defer func() { _ = os.Remove(configFile) }()

	content, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}

	if strings.Contains(string(content), WorkspaceModelsNodeMountPath) {
		t.Fatalf("expected config to omit workspace models mount, got:\n%s", content)
	}
}

func TestCreateClusterConfigWithWorkspaceModelsMount(t *testing.T) {
	cluster := NewKindCluster("unit-test", false)
	workspaceModelsDir := t.TempDir()
	cluster.SetWorkspaceModelsDir(workspaceModelsDir)

	configFile, err := cluster.createClusterConfig()
	if err != nil {
		t.Fatalf("createClusterConfig returned error: %v", err)
	}
	defer func() { _ = os.Remove(configFile) }()

	content, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}

	config := string(content)
	if count := strings.Count(config, WorkspaceModelsNodeMountPath); count != 2 {
		t.Fatalf("expected workspace models mount on both kind nodes, got %d occurrences in:\n%s", count, config)
	}
	if !strings.Contains(config, workspaceModelsDir) {
		t.Fatalf("expected config to include workspace models host path %q, got:\n%s", workspaceModelsDir, config)
	}
}
