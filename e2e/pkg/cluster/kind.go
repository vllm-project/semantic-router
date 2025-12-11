package cluster

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// KindCluster manages Kind cluster lifecycle
type KindCluster struct {
	Name    string
	Verbose bool
}

// NewKindCluster creates a new Kind cluster manager
func NewKindCluster(name string, verbose bool) *KindCluster {
	return &KindCluster{
		Name:    name,
		Verbose: verbose,
	}
}

// Create creates a new Kind cluster
func (k *KindCluster) Create(ctx context.Context) error {
	k.log("Creating Kind cluster: %s", k.Name)

	// Check if cluster already exists
	exists, err := k.Exists(ctx)
	if err != nil {
		return fmt.Errorf("failed to check if cluster exists: %w", err)
	}

	if exists {
		k.log("Cluster %s already exists", k.Name)
		return nil
	}

	// Mount /mnt from host into Kind node so storage provisioner can use it (more disk space)
	configContent := fmt.Sprintf(`kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: %s
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
  - role: worker
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
`, k.Name)

	configFile, err := os.CreateTemp("", "kind-config-*.yaml")
	if err != nil {
		return fmt.Errorf("failed to create temp config file: %w", err)
	}
	defer os.Remove(configFile.Name())

	if _, err := configFile.WriteString(configContent); err != nil {
		configFile.Close()
		return fmt.Errorf("failed to write config file: %w", err)
	}
	configFile.Close()

	k.log("Using Kind config with /mnt mount for storage")

	// Create cluster with config file
	cmd := exec.CommandContext(ctx, "kind", "create", "cluster", "--name", k.Name, "--config", configFile.Name())
	if k.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create cluster: %w", err)
	}

	// Wait for cluster to be ready
	k.log("Waiting for cluster to be ready...")
	if err := k.WaitForReady(ctx, 5*time.Minute); err != nil {
		return fmt.Errorf("cluster failed to become ready: %w", err)
	}

	// Configure storage provisioner to use /mnt (75GB) instead of /tmp (limited space)
	kubeConfig, err := k.GetKubeConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to get kubeconfig: %w", err)
	}
	defer os.Remove(kubeConfig)

	// Simple one-liner: update ConfigMap and restart provisioner
	// Models downloaded in pods will be stored in /mnt via PVCs
	exec.CommandContext(ctx, "kubectl", "--kubeconfig", kubeConfig,
		"patch", "configmap", "local-path-config", "-n", "local-path-storage",
		"--type", "merge",
		"-p", `{"data":{"config.json":"{\"nodePathMap\":[{\"node\":\"DEFAULT_PATH_FOR_NON_LISTED_NODES\",\"paths\":[\"/mnt/local-path-provisioner\"]}]}"}}`).Run()
	exec.CommandContext(ctx, "kubectl", "--kubeconfig", kubeConfig,
		"rollout", "restart", "deployment/local-path-provisioner", "-n", "local-path-storage").Run()

	k.log("Cluster %s created successfully", k.Name)
	return nil
}

// Delete deletes the Kind cluster
func (k *KindCluster) Delete(ctx context.Context) error {
	k.log("Deleting Kind cluster: %s", k.Name)

	cmd := exec.CommandContext(ctx, "kind", "delete", "cluster", "--name", k.Name)
	if k.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete cluster: %w", err)
	}

	k.log("Cluster %s deleted successfully", k.Name)
	return nil
}

// Exists checks if the cluster exists
func (k *KindCluster) Exists(ctx context.Context) (bool, error) {
	cmd := exec.CommandContext(ctx, "kind", "get", "clusters")
	output, err := cmd.Output()
	if err != nil {
		return false, fmt.Errorf("failed to list clusters: %w", err)
	}

	clusters := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, cluster := range clusters {
		if cluster == k.Name {
			return true, nil
		}
	}

	return false, nil
}

// WaitForReady waits for the cluster to be ready
func (k *KindCluster) WaitForReady(ctx context.Context, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "kubectl", "wait",
		"--for=condition=Ready",
		"nodes",
		"--all",
		"--timeout=300s")

	if k.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("nodes failed to become ready: %w", err)
	}

	return nil
}

// GetKubeConfig returns the path to the kubeconfig file
func (k *KindCluster) GetKubeConfig(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, "kind", "get", "kubeconfig", "--name", k.Name)
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get kubeconfig: %w", err)
	}

	// Write kubeconfig to temp file
	tmpFile, err := os.CreateTemp("", fmt.Sprintf("kubeconfig-%s-*.yaml", k.Name))
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := tmpFile.Write(output); err != nil {
		tmpFile.Close()
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to write kubeconfig: %w", err)
	}

	tmpFile.Close()
	return tmpFile.Name(), nil
}

func (k *KindCluster) log(format string, args ...interface{}) {
	if k.Verbose {
		fmt.Printf("[Kind] "+format+"\n", args...)
	}
}
