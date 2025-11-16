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

	// Create cluster
	cmd := exec.CommandContext(ctx, "kind", "create", "cluster", "--name", k.Name)
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
