package cluster

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReadinessUsesOnlyTheRequestedClustersKubeconfig(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	kind := "#!/bin/sh\n[ \"$*\" = 'get kubeconfig --name isolated' ] || exit 7\nprintf 'isolated-config'\n"
	kubectl := "#!/bin/sh\n[ \"$1\" = '--kubeconfig' ] || exit 8\n[ \"$(cat \"$2\")\" = 'isolated-config' ] || exit 9\nprintf '%s' \"$2\" > \"$KIND_TEST_LOG\"\n"
	for name, program := range map[string]string{"kind": kind, "kubectl": kubectl} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(program), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("KIND_TEST_LOG", log)
	t.Setenv("KUBECONFIG", filepath.Join(dir, "unrelated-config"))
	if err := NewKindCluster("isolated", false).WaitForReady(context.Background(), 10*time.Second); err != nil {
		t.Fatal(err)
	}
	config, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(string(config)); !os.IsNotExist(err) {
		t.Fatalf("temporary kubeconfig was not removed: %v", err)
	}
}

func TestExistingClusterCannotBecomeOwnedOrBeDeleted(t *testing.T) {
	dir := t.TempDir()
	commandLog := filepath.Join(dir, "calls")
	program := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$KIND_TEST_LOG\"\nif [ \"$1 $2\" = 'get clusters' ]; then echo existing; fi\n"
	if err := os.WriteFile(filepath.Join(dir, "kind"), []byte(program), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("KIND_TEST_LOG", commandLog)
	c := NewKindCluster("existing", false)
	if err := c.Create(context.Background()); err == nil {
		t.Fatal("silently reused existing cluster")
	}
	if c.Created {
		t.Fatal("adopted somebody else's cluster")
	}
	if err := c.Delete(context.Background()); err == nil {
		t.Fatal("unowned cluster deletion succeeded")
	}
	log, err := os.ReadFile(commandLog)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(log), "delete") {
		t.Fatalf("issued destructive command: %s", log)
	}
}

func TestClusterModelStagingIsNotShared(t *testing.T) {
	a, b := NewKindCluster("run-a", false), NewKindCluster("run-b", false)
	if a.ModelsDir() == b.ModelsDir() {
		t.Fatal("cluster model staging is shared")
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
