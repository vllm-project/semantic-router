package cluster

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBatchClustersIsolateStorageAndModelMounts(t *testing.T) {
	previous := ""
	for _, name := range []string{"first", "second"} {
		state := t.TempDir()
		storage := filepath.Join(state, "storage")
		models := filepath.Join(state, "models")
		t.Setenv("E2E_KIND_STORAGE_DIR", storage)
		t.Setenv("E2E_KIND_MODELS_DIR", models)
		cluster := NewKindCluster(name, false)
		configFile, err := cluster.createClusterConfig()
		if err != nil {
			t.Fatal(err)
		}
		content, err := os.ReadFile(configFile)
		_ = os.Remove(configFile)
		if err != nil {
			t.Fatal(err)
		}
		config := string(content)
		for _, path := range []string{storage, models} {
			if strings.Count(config, "hostPath: "+path+"\n") != 2 {
				t.Fatalf("both nodes must mount private directory %s: %s", path, config)
			}
		}
		if previous != "" && strings.Contains(config, previous) {
			t.Fatal("cluster inherited another profile's storage")
		}
		previous = state
	}
}

func TestBatchClusterRejectsRelativeHostPaths(t *testing.T) {
	for _, variable := range []string{"E2E_KIND_STORAGE_DIR", "E2E_KIND_MODELS_DIR"} {
		t.Run(variable, func(t *testing.T) {
			t.Setenv(variable, "relative/path")
			if _, err := NewKindCluster("test", false).createClusterConfig(); err == nil {
				t.Fatal("relative batch storage must be rejected")
			}
		})
	}
}

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

func TestWorkspaceModelsMountDoesNotOverlapKindStorageMount(t *testing.T) {
	storagePath := strings.TrimSuffix(kindStorageNodeMountPath, "/") + "/"
	modelsPath := strings.TrimSuffix(WorkspaceModelsNodeMountPath, "/") + "/"
	if strings.HasPrefix(modelsPath, storagePath) || strings.HasPrefix(storagePath, modelsPath) {
		t.Fatalf(
			"workspace models mount %q must not overlap kind storage mount %q",
			WorkspaceModelsNodeMountPath,
			kindStorageNodeMountPath,
		)
	}
}

func TestCreateClusterArgsUsesPinnedNodeImage(t *testing.T) {
	t.Setenv("KIND_NODE_IMAGE", "kindest/node:test@sha256:abc")
	cluster := NewKindCluster("unit-test", false)

	args := cluster.createClusterArgs("/tmp/kind-config.yaml")

	expected := []string{
		"create", "cluster", "--name", "unit-test",
		"--image", "kindest/node:test@sha256:abc",
		"--config", "/tmp/kind-config.yaml",
	}
	if strings.Join(args, " ") != strings.Join(expected, " ") {
		t.Fatalf("createClusterArgs returned %v, want %v", args, expected)
	}
}

func TestCreateClusterArgsKeepsDefaultNodeImageWhenUnset(t *testing.T) {
	t.Setenv("KIND_NODE_IMAGE", "")
	cluster := NewKindCluster("unit-test", false)

	args := cluster.createClusterArgs("/tmp/kind-config.yaml")

	if strings.Contains(strings.Join(args, " "), " --image ") {
		t.Fatalf("createClusterArgs unexpectedly selected an image: %v", args)
	}
}
