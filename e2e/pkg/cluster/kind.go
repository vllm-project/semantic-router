package cluster

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// KindCluster manages Kind cluster lifecycle
type KindCluster struct {
	Name       string
	Verbose    bool
	GPUEnabled bool // Enable GPU support for the cluster
}

// NewKindCluster creates a new Kind cluster manager
func NewKindCluster(name string, verbose bool) *KindCluster {
	return &KindCluster{
		Name:    name,
		Verbose: verbose,
	}
}

// SetGPUEnabled enables GPU support for the cluster
func (k *KindCluster) SetGPUEnabled(enabled bool) {
	k.GPUEnabled = enabled
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

	// If GPU enabled, verify Docker nvidia runtime first
	if k.GPUEnabled {
		if err := k.verifyNvidiaRuntime(ctx); err != nil {
			return err
		}
	}

	// Create cluster config with /mnt mount for storage (and GPU support if enabled)
	var cmd *exec.Cmd
	configFile, err := k.createClusterConfig()
	if err != nil {
		return fmt.Errorf("failed to create cluster config: %w", err)
	}
	defer os.Remove(configFile)

	if k.GPUEnabled {
		k.log("Creating cluster with GPU support and /mnt mount for storage...")
		cmd = exec.CommandContext(ctx, "kind", "create", "cluster",
			"--name", k.Name,
			"--config", configFile,
			"--wait", "5m")
	} else {
		k.log("Using Kind config with /mnt mount for storage")
		cmd = exec.CommandContext(ctx, "kind", "create", "cluster",
			"--name", k.Name,
			"--config", configFile)
	}
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

	// If GPU enabled, setup NVIDIA libraries
	if k.GPUEnabled {
		if err := k.setupGPULibraries(ctx); err != nil {
			return fmt.Errorf("failed to setup GPU libraries: %w", err)
		}
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

// verifyNvidiaRuntime checks if Docker's default runtime is nvidia
func (k *KindCluster) verifyNvidiaRuntime(ctx context.Context) error {
	k.log("Verifying Docker nvidia runtime...")
	cmd := exec.CommandContext(ctx, "docker", "info")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get docker info: %w", err)
	}

	if !strings.Contains(string(output), "Default Runtime: nvidia") {
		k.log("ERROR: Docker default runtime is not nvidia!")
		k.log("Run: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default")
		k.log("Then restart Docker: sudo systemctl restart docker")
		return fmt.Errorf("docker default runtime must be nvidia for GPU support")
	}
	k.log("✅ Docker default runtime is nvidia")
	return nil
}

// getHostMountPath returns the appropriate host path for mounting based on OS
// On Linux: uses /mnt (standard location)
// On macOS: creates a temporary directory in /tmp (Docker Desktop compatible)
// On Windows: creates a temporary directory in user's temp folder
func (k *KindCluster) getHostMountPath() (string, error) {
	switch runtime.GOOS {
	case "linux":
		// On Linux, use /mnt as it's standard and typically has more space
		return "/mnt", nil
	case "darwin":
		// On macOS, Docker Desktop only allows mounting from specific locations
		// Use /tmp which is allowed by default
		tmpDir := filepath.Join(os.TempDir(), "kind-mnt-"+k.Name)
		if err := os.MkdirAll(tmpDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create temp mount directory: %w", err)
		}
		k.log("Using macOS-compatible mount path: %s", tmpDir)
		return tmpDir, nil
	case "windows":
		// On Windows, use temp directory
		tmpDir := filepath.Join(os.TempDir(), "kind-mnt-"+k.Name)
		if err := os.MkdirAll(tmpDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create temp mount directory: %w", err)
		}
		k.log("Using Windows-compatible mount path: %s", tmpDir)
		return tmpDir, nil
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
}

// createClusterConfig creates a Kind config file with host mount for storage
// and optionally GPU support if GPUEnabled is true
func (k *KindCluster) createClusterConfig() (string, error) {
	// Get OS-appropriate host path for mounting
	hostPath, err := k.getHostMountPath()
	if err != nil {
		return "", err
	}

	// Base config with host mount for storage (always included)
	kindConfig := fmt.Sprintf(`kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: %s
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: %s
        containerPath: /mnt`, k.Name, hostPath)

	// Add GPU mount to worker if GPU is enabled
	if k.GPUEnabled {
		kindConfig += fmt.Sprintf(`
  - role: worker
    extraMounts:
      - hostPath: %s
        containerPath: /mnt
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
`, hostPath)
	} else {
		kindConfig += fmt.Sprintf(`
  - role: worker
    extraMounts:
      - hostPath: %s
        containerPath: /mnt
`, hostPath)
	}

	configFile, err := os.CreateTemp("", "kind-config-*.yaml")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := configFile.WriteString(kindConfig); err != nil {
		configFile.Close()
		os.Remove(configFile.Name())
		return "", fmt.Errorf("failed to write config: %w", err)
	}
	configFile.Close()

	return configFile.Name(), nil
}

// setupGPULibraries copies NVIDIA libraries to the Kind worker
func (k *KindCluster) setupGPULibraries(ctx context.Context) error {
	workerName := k.Name + "-worker"

	// Get driver version (same as script: nvidia-smi ... | head -1)
	k.log("Detecting NVIDIA driver version...")
	driverCmd := exec.CommandContext(ctx, "bash", "-c", "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
	driverOutput, err := driverCmd.Output()
	if err != nil {
		k.log("nvidia-smi not available, skipping GPU library setup")
		return nil
	}
	driverVersion := strings.TrimSpace(string(driverOutput))
	// Remove any extra newlines/spaces
	driverVersion = strings.Split(driverVersion, "\n")[0]
	k.log("Detected NVIDIA driver version: %s", driverVersion)

	// Verify GPU devices exist in worker
	checkGPU := exec.CommandContext(ctx, "docker", "exec", workerName, "ls", "/dev/nvidia0")
	if err := checkGPU.Run(); err != nil {
		return fmt.Errorf("GPU devices not found in Kind worker - cluster may not have GPU support")
	}
	k.log("✅ GPU devices found in Kind worker")

	// Check if libraries already exist
	checkLibs := exec.CommandContext(ctx, "docker", "exec", workerName, "ls", "/nvidia-driver-libs/nvidia-smi")
	if checkLibs.Run() == nil {
		k.log("GPU libraries already set up")
		return k.deployDevicePlugin(ctx)
	}

	k.log("Setting up NVIDIA libraries in Kind worker...")

	// Create directory
	mkdirCmd := exec.CommandContext(ctx, "docker", "exec", workerName, "mkdir", "-p", "/nvidia-driver-libs")
	if err := mkdirCmd.Run(); err != nil {
		return fmt.Errorf("failed to create nvidia-driver-libs directory: %w", err)
	}

	// Copy nvidia-smi
	copyNvidiaSmi := exec.CommandContext(ctx, "bash", "-c",
		fmt.Sprintf("tar -cf - -C /usr/bin nvidia-smi | docker exec -i %s tar -xf - -C /nvidia-driver-libs/", workerName))
	if err := copyNvidiaSmi.Run(); err != nil {
		return fmt.Errorf("failed to copy nvidia-smi: %w", err)
	}

	// Copy NVIDIA libraries (same as all-in-one script from docs)
	k.log("Copying NVIDIA libraries from /usr/lib64...")
	copyLibsScript := "tar -cf - -C /usr/lib64 libnvidia-ml.so." + driverVersion + " libcuda.so." + driverVersion + " | docker exec -i " + workerName + " tar -xf - -C /nvidia-driver-libs/"
	copyLibs := exec.CommandContext(ctx, "bash", "-c", copyLibsScript)
	if k.Verbose {
		k.log("Running: %s", copyLibsScript)
	}
	if output, err := copyLibs.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to copy NVIDIA libraries: %w\nOutput: %s", err, string(output))
	}

	// Create symlinks
	symlinkCmd := exec.CommandContext(ctx, "docker", "exec", workerName, "bash", "-c",
		fmt.Sprintf("cd /nvidia-driver-libs && ln -sf libnvidia-ml.so.%s libnvidia-ml.so.1 && ln -sf libcuda.so.%s libcuda.so.1 && chmod +x nvidia-smi",
			driverVersion, driverVersion))
	if err := symlinkCmd.Run(); err != nil {
		return fmt.Errorf("failed to create symlinks: %w", err)
	}

	// Verify nvidia-smi works
	verifyCmd := exec.CommandContext(ctx, "docker", "exec", workerName, "bash", "-c",
		"LD_LIBRARY_PATH=/nvidia-driver-libs /nvidia-driver-libs/nvidia-smi")
	if output, err := verifyCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("nvidia-smi verification failed: %w\nOutput: %s", err, string(output))
	}
	k.log("✅ nvidia-smi verified in Kind worker")

	// Deploy device plugin
	return k.deployDevicePlugin(ctx)
}

// deployDevicePlugin deploys the NVIDIA device plugin
func (k *KindCluster) deployDevicePlugin(ctx context.Context) error {
	// Check if already deployed
	checkCmd := exec.CommandContext(ctx, "kubectl", "get", "daemonset",
		"nvidia-device-plugin-daemonset", "-n", "kube-system")
	if checkCmd.Run() == nil {
		k.log("NVIDIA device plugin already deployed")
		return nil
	}

	k.log("Deploying NVIDIA device plugin...")

	devicePluginYAML := `apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.14.1
        name: nvidia-device-plugin-ctr
        env:
        - name: LD_LIBRARY_PATH
          value: "/nvidia-driver-libs"
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
        - name: nvidia-driver-libs
          mountPath: /nvidia-driver-libs
          readOnly: true
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: dev
        hostPath:
          path: /dev
      - name: nvidia-driver-libs
        hostPath:
          path: /nvidia-driver-libs`

	tmpFile, err := os.CreateTemp("", "nvidia-device-plugin-*.yaml")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(devicePluginYAML); err != nil {
		return fmt.Errorf("failed to write device plugin manifest: %w", err)
	}
	tmpFile.Close()

	applyCmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", tmpFile.Name())
	if output, err := applyCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to apply device plugin: %w\nOutput: %s", err, string(output))
	}

	k.log("NVIDIA device plugin deployed, waiting for it to be ready...")
	time.Sleep(20 * time.Second)

	// Verify GPUs are allocatable
	verifyCmd := exec.CommandContext(ctx, "kubectl", "get", "nodes",
		"-o", "custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu")
	if output, err := verifyCmd.CombinedOutput(); err != nil {
		k.log("Warning: Could not verify GPU allocatable: %v", err)
	} else {
		k.log("GPU allocatable status:\n%s", string(output))
	}

	k.log("✅ GPU setup complete")
	return nil
}
