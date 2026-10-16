package cluster

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	kindStorageNodeMountPath     = "/mnt"
	WorkspaceModelsNodeMountPath = "/opt/semantic-router/workspace-models"

	// kindClusterBootstrapAttempts allows one retry after a failed bootstrap.
	// Bootstrapping reaches the Docker daemon and the node image registry, so a
	// single transient failure must not fail a whole CI profile.
	kindClusterBootstrapAttempts = 2

	// kindClusterBootstrapDelay gives Docker time to release the deleted
	// cluster's containers and networks before the retry starts.
	kindClusterBootstrapDelay = 30 * time.Second

	// kindClusterReadyTimeout bounds how long the nodes may take to become Ready.
	kindClusterReadyTimeout = 5 * time.Minute
)

// KindCluster manages Kind cluster lifecycle
type KindCluster struct {
	Name               string
	Verbose            bool
	GPUEnabled         bool // Enable GPU support for the cluster
	WorkspaceModelsDir string

	// bootstrapDelay is the pause between bootstrap attempts. It is a field so
	// tests can drop the wait instead of sleeping in a unit test.
	bootstrapDelay time.Duration
}

// NewKindCluster creates a new Kind cluster manager
func NewKindCluster(name string, verbose bool) *KindCluster {
	return &KindCluster{
		Name:           name,
		Verbose:        verbose,
		bootstrapDelay: kindClusterBootstrapDelay,
	}
}

// SetGPUEnabled enables GPU support for the cluster
func (k *KindCluster) SetGPUEnabled(enabled bool) {
	k.GPUEnabled = enabled
}

// SetWorkspaceModelsDir enables an opt-in host mount for the workspace models directory.
func (k *KindCluster) SetWorkspaceModelsDir(dir string) {
	k.WorkspaceModelsDir = dir
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

	configFile, err := k.createClusterConfig()
	if err != nil {
		return fmt.Errorf("failed to create cluster config: %w", err)
	}
	defer removeFile(configFile)

	if err := k.bootstrap(ctx, configFile); err != nil {
		return err
	}

	// Configure storage provisioner to use /mnt (75GB) instead of /tmp (limited space)
	if err := k.configureStorageProvisioner(ctx); err != nil {
		return err
	}

	// If GPU enabled, setup NVIDIA libraries
	if k.GPUEnabled {
		if err := k.setupGPULibraries(ctx); err != nil {
			return fmt.Errorf("failed to setup GPU libraries: %w", err)
		}
	}

	k.log("Cluster %s created successfully", k.Name)
	return nil
}

// bootstrap creates the cluster and waits for its nodes to become Ready. A
// failed attempt leaves a partial cluster behind, so the retry deletes it first
// and then waits before trying again.
func (k *KindCluster) bootstrap(ctx context.Context, configFile string) error {
	var lastErr error
	for attempt := 1; attempt <= kindClusterBootstrapAttempts; attempt++ {
		if attempt > 1 {
			k.log(
				"Retrying Kind cluster bootstrap (attempt %d/%d)",
				attempt, kindClusterBootstrapAttempts,
			)
			k.deleteBestEffort(ctx)
			if err := sleep(ctx, k.bootstrapDelay); err != nil {
				return errors.Join(err, lastErr)
			}
		}

		lastErr = k.createAndWait(ctx, configFile)
		if lastErr == nil {
			return nil
		}
		k.log(
			"Kind cluster bootstrap attempt %d/%d failed: %v",
			attempt, kindClusterBootstrapAttempts, lastErr,
		)
	}
	return lastErr
}

// createAndWait runs `kind create cluster` and waits for the nodes to be Ready.
func (k *KindCluster) createAndWait(ctx context.Context, configFile string) error {
	if err := k.runCreateClusterCommand(ctx, configFile); err != nil {
		return fmt.Errorf("failed to create cluster: %w", err)
	}
	k.allowPodARP(ctx)

	k.log("Waiting for cluster to be ready...")
	if err := k.WaitForReady(ctx, kindClusterReadyTimeout); err != nil {
		return fmt.Errorf("cluster failed to become ready: %w", err)
	}
	return nil
}

// deleteBestEffort removes the cluster before a retry. The cluster may not exist
// at all, which is not an error here.
func (k *KindCluster) deleteBestEffort(ctx context.Context) {
	if err := k.Delete(ctx); err != nil {
		k.log("Warning: could not delete cluster before retry: %v", err)
	}
}

// sleep waits for the delay, or returns early when the context is cancelled.
func sleep(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (k *KindCluster) runCreateClusterCommand(ctx context.Context, configFile string) error {
	args := k.createClusterArgs(configFile)
	cmd := exec.CommandContext(ctx, "kind", args...)
	if k.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (k *KindCluster) createClusterArgs(configFile string) []string {
	args := []string{"create", "cluster", "--name", k.Name}
	if nodeImage := strings.TrimSpace(os.Getenv("KIND_NODE_IMAGE")); nodeImage != "" {
		args = append(args, "--image", nodeImage)
	}
	args = append(args, "--config", configFile)
	if k.GPUEnabled {
		k.log("Creating cluster with GPU support and /mnt mount for storage...")
		args = append(args, "--wait", "5m")
	} else {
		k.log("Using Kind config with /mnt mount for storage")
	}
	return args
}

// podARPScript lets every interface of a node answer ARP for its own /32.
// Hosts that set net.ipv4.conf.default.arp_ignore=2 pass it to Kind's nodes;
// Kind resets only net.ipv4.conf.all, but an interface's effective value is
// the larger of the two, so kindnet's /32 veths would ignore their pods' ARP
// requests and every pod would lose its network.
const podARPScript = `sysctl -qw net.ipv4.conf.default.arp_ignore=0 && for setting in /proc/sys/net/ipv4/conf/*/arp_ignore; do echo 0 > "$setting"; done`

func (k *KindCluster) allowPodARP(ctx context.Context) {
	output, err := exec.CommandContext(ctx, "kind", "get", "nodes", "--name", k.Name).Output() //nolint:gosec // The cluster name comes from the E2E run, never from a request.
	if err != nil {
		k.log("Warning: list Kind nodes: %v", err)
		return
	}
	for _, node := range strings.Fields(string(output)) {
		if err := exec.CommandContext(ctx, "docker", "exec", node, "sh", "-c", podARPScript).Run(); err != nil {
			k.log("Warning: reset arp_ignore on %s: %v", node, err)
		}
	}
}

func (k *KindCluster) configureStorageProvisioner(ctx context.Context) error {
	kubeConfig, err := k.GetKubeConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to get kubeconfig: %w", err)
	}
	defer removeFile(kubeConfig)

	k.runBestEffortKubectl(
		ctx,
		kubeConfig,
		"patch", "configmap", "local-path-config", "-n", "local-path-storage",
		"--type", "merge",
		"-p", `{"data":{"config.json":"{\"nodePathMap\":[{\"node\":\"DEFAULT_PATH_FOR_NON_LISTED_NODES\",\"paths\":[\"/mnt/local-path-provisioner\"]}]}"}}`,
	)
	k.runBestEffortKubectl(
		ctx,
		kubeConfig,
		"rollout", "restart", "deployment/local-path-provisioner", "-n", "local-path-storage",
	)
	return nil
}

func (k *KindCluster) runBestEffortKubectl(ctx context.Context, kubeConfig string, args ...string) {
	cmdArgs := append([]string{"--kubeconfig", kubeConfig}, args...)
	cmd := exec.CommandContext(ctx, "kubectl", cmdArgs...)
	if err := cmd.Run(); err != nil {
		k.log("Warning: kubectl %s failed: %v", strings.Join(args, " "), err)
	}
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
		closeFile(tmpFile)
		removeFile(tmpFile.Name())
		return "", fmt.Errorf("failed to write kubeconfig: %w", err)
	}

	closeFile(tmpFile)
	return tmpFile.Name(), nil
}

func (k *KindCluster) log(format string, args ...interface{}) {
	if k.Verbose {
		fmt.Printf("[Kind] "+format+"\n", args...)
	}
}

func closeFile(file *os.File) {
	_ = file.Close()
}

func removeFile(path string) {
	_ = os.Remove(path)
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
	if directory := strings.TrimSpace(os.Getenv("E2E_KIND_STORAGE_DIR")); directory != "" {
		if !filepath.IsAbs(directory) {
			return "", fmt.Errorf("E2E_KIND_STORAGE_DIR must be an absolute path")
		}
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return "", fmt.Errorf("create isolated Kind storage: %w", err)
		}
		return directory, nil
	}
	switch runtime.GOOS {
	case "linux":
		// On Linux, use /mnt as it's standard and typically has more space
		return "/mnt", nil
	case "darwin":
		// On macOS, Docker Desktop only allows mounting from specific locations
		// Use /tmp which is allowed by default
		tmpDir := filepath.Join(os.TempDir(), "kind-mnt-"+k.Name)
		if err := os.MkdirAll(tmpDir, 0o755); err != nil {
			return "", fmt.Errorf("failed to create temp mount directory: %w", err)
		}
		k.log("Using macOS-compatible mount path: %s", tmpDir)
		return tmpDir, nil
	case "windows":
		// On Windows, use temp directory
		tmpDir := filepath.Join(os.TempDir(), "kind-mnt-"+k.Name)
		if err := os.MkdirAll(tmpDir, 0o755); err != nil {
			return "", fmt.Errorf("failed to create temp mount directory: %w", err)
		}
		k.log("Using Windows-compatible mount path: %s", tmpDir)
		return tmpDir, nil
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
}

// MLModelsHostPath identifies the host directory mounted into this profile's nodes.
func MLModelsHostPath() string {
	if directory := strings.TrimSpace(os.Getenv("E2E_KIND_MODELS_DIR")); directory != "" {
		return directory
	}
	return "/tmp/kind-ml-models"
}

// createClusterConfig creates a Kind config file with host mount for storage
// and optionally GPU support if GPUEnabled is true
func (k *KindCluster) createClusterConfig() (string, error) {
	// Get OS-appropriate host path for mounting
	hostPath, err := k.getHostMountPath()
	if err != nil {
		return "", err
	}

	// Ensure ML models mount directory exists BEFORE Kind cluster creation
	// This is required because Kind mounts are set up at cluster creation time
	mlModelsDir := MLModelsHostPath()
	if !filepath.IsAbs(mlModelsDir) {
		return "", fmt.Errorf("E2E_KIND_MODELS_DIR must be an absolute path")
	}
	if mkdirErr := os.MkdirAll(mlModelsDir, 0o755); mkdirErr != nil {
		k.log("Warning: failed to create ML models directory %s: %v", mlModelsDir, mkdirErr)
	}

	workspaceModelsMount := ""
	if k.WorkspaceModelsDir != "" {
		if mkdirErr := os.MkdirAll(k.WorkspaceModelsDir, 0o755); mkdirErr != nil {
			return "", fmt.Errorf("failed to create workspace models directory %s: %w", k.WorkspaceModelsDir, mkdirErr)
		}
		workspaceModelsMount = fmt.Sprintf(`
      - hostPath: %s
        containerPath: %s`, k.WorkspaceModelsDir, WorkspaceModelsNodeMountPath)
	}

	// Base config with host mount for storage (always included)
	// Also mount /tmp/kind-ml-models for ML model selection E2E tests
	kindConfig := fmt.Sprintf(`kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: %s
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: %s
        containerPath: %s
      - hostPath: %s
        containerPath: /tmp/ml-models%s`, k.Name, hostPath, kindStorageNodeMountPath, mlModelsDir, workspaceModelsMount)

	// Add GPU mount to worker if GPU is enabled
	if k.GPUEnabled {
		kindConfig += fmt.Sprintf(`
  - role: worker
    extraMounts:
      - hostPath: %s
        containerPath: %s
      - hostPath: %s
        containerPath: /tmp/ml-models%s
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
`, hostPath, kindStorageNodeMountPath, mlModelsDir, workspaceModelsMount)
	} else {
		kindConfig += fmt.Sprintf(`
  - role: worker
    extraMounts:
      - hostPath: %s
        containerPath: %s
      - hostPath: %s
        containerPath: /tmp/ml-models%s
`, hostPath, kindStorageNodeMountPath, mlModelsDir, workspaceModelsMount)
	}

	configFile, err := os.CreateTemp("", "kind-config-*.yaml")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	if _, err := configFile.WriteString(kindConfig); err != nil {
		closeFile(configFile)
		removeFile(configFile.Name())
		return "", fmt.Errorf("failed to write config: %w", err)
	}
	closeFile(configFile)

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
	defer removeFile(tmpFile.Name())

	if _, err := tmpFile.WriteString(devicePluginYAML); err != nil {
		return fmt.Errorf("failed to write device plugin manifest: %w", err)
	}
	closeFile(tmpFile)

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
