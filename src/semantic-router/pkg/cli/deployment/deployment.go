package deployment

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// getPIDFilePath returns the cross-platform PID file path
// NOTE: Uses os.Getuid() which returns -1 on Windows for all users.
// Focus is on Linux/macOS. For future Windows support, consider using
// user.Current().Username or os.UserHomeDir() for true cross-platform
// user isolation.
func getPIDFilePath() string {
	return filepath.Join(os.TempDir(), fmt.Sprintf("vsr-local-deployment-%d.pid", os.Getuid()))
}

// getLogFilePath returns the cross-platform log file path
// NOTE: See getPIDFilePath for Windows limitation regarding os.Getuid().
func getLogFilePath() string {
	return filepath.Join(os.TempDir(), fmt.Sprintf("vsr-local-deployment-%d.log", os.Getuid()))
}

// isProcessRunning checks if a process with the given PID is still running
func isProcessRunning(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// Send signal 0 to check if process exists (doesn't actually signal)
	// This is Unix-specific but works on Linux/macOS
	err = process.Signal(syscall.Signal(0))
	return err == nil
}

// stopProcess stops a process gracefully (SIGTERM then SIGKILL if needed)
func stopProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("process not found: %w", err)
	}

	// Send SIGTERM for graceful shutdown
	if err := process.Signal(syscall.SIGTERM); err != nil {
		return fmt.Errorf("failed to send SIGTERM: %w", err)
	}

	// Wait up to 10 seconds for graceful shutdown
	for i := 0; i < 10; i++ {
		time.Sleep(1 * time.Second)
		if err := process.Signal(syscall.Signal(0)); err != nil {
			// Process is gone
			return nil
		}
	}

	// Still running, send SIGKILL
	if err := process.Kill(); err != nil {
		return fmt.Errorf("failed to kill process: %w", err)
	}

	time.Sleep(1 * time.Second)
	return nil
}

// DeploymentStatus represents the status of a deployment
type DeploymentStatus struct {
	Type        string
	IsRunning   bool
	ReleaseName string
	Namespace   string
	Components  []ComponentStatus
	Endpoints   []string
	Uptime      string
}

// ComponentStatus represents the status of a component
type ComponentStatus struct {
	Name    string
	Status  string
	Message string
}

// DeployLocal deploys the router as a local process
// If force is true, any existing router process will be stopped before starting a new one.
func DeployLocal(configPath string, force bool) error {
	cli.Info("Deploying router locally...")

	// Check if binary exists
	binPath := "bin/router"
	if _, err := os.Stat(binPath); os.IsNotExist(err) {
		cli.Warning("Router binary not found. Building...")
		if err := buildRouter(); err != nil {
			return fmt.Errorf("failed to build router: %w", err)
		}
	}

	// Get absolute config path
	absConfigPath, err := filepath.Abs(configPath)
	if err != nil {
		return fmt.Errorf("failed to resolve config path: %w", err)
	}

	cli.Info(fmt.Sprintf("Starting router with config: %s", absConfigPath))

	// Get cross-platform file paths
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Check if router is already running
	if _, errShadow := os.Stat(pidFilePath); errShadow == nil {
		// PID file exists, check if process is actually running
		pidBytes, readErr := os.ReadFile(pidFilePath)
		if readErr == nil {
			if pid, atoiErr := strconv.Atoi(strings.TrimSpace(string(pidBytes))); atoiErr == nil {
				if isProcessRunning(pid) {
					// Process is running
					if !force {
						// Error by default
						cli.Warning(fmt.Sprintf("Router is already running (PID: %d)", pid))
						cli.Info(fmt.Sprintf("PID file: %s", pidFilePath))
						cli.Info(fmt.Sprintf("Log file: %s", logFilePath))
						cli.Info("")
						cli.Info("Options:")
						cli.Info("  1. Stop existing: vsr undeploy local")
						cli.Info("  2. Force replace: vsr deploy local --force")
						return fmt.Errorf("router already running (use --force to replace)")
					}

					// --force flag provided, stop existing process
					cli.Warning(fmt.Sprintf("Stopping existing router (PID: %d)...", pid))
					if stopErr := stopProcess(pid); stopErr != nil {
						cli.Warning(fmt.Sprintf("Failed to stop existing process: %v", stopErr))
						cli.Info("Continuing anyway (process may have already stopped)")
					} else {
						cli.Success("Existing router stopped")
					}

					// Clean up PID file
					os.Remove(pidFilePath)
				} else {
					// PID file exists but process is not running (stale file)
					cli.Warning(fmt.Sprintf("Found stale PID file (process %d not running)", pid))
					cli.Info("Cleaning up stale PID file...")
					os.Remove(pidFilePath)
				}
			}
		}
	}

	// Open log file for output (Issue #5: restrictive permissions 0600)
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("failed to create log file: %w", err)
	}
	defer logFile.Close()

	// Start router process
	cmd := exec.Command(binPath, "--config", absConfigPath)
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	// Set library path for candle binding (Rust FFI library)
	// Get current working directory to construct the library path
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get working directory: %w", err)
	}
	candleLibPath := filepath.Join(cwd, "candle-binding", "target", "release")

	// Inherit current environment and add/update library path
	// Linux uses LD_LIBRARY_PATH, macOS uses DYLD_LIBRARY_PATH
	cmd.Env = os.Environ()

	// Set LD_LIBRARY_PATH for Linux
	existingLDPath := os.Getenv("LD_LIBRARY_PATH")
	if existingLDPath != "" {
		cmd.Env = append(cmd.Env, fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", candleLibPath, existingLDPath))
	} else {
		cmd.Env = append(cmd.Env, fmt.Sprintf("LD_LIBRARY_PATH=%s", candleLibPath))
	}

	// Set DYLD_LIBRARY_PATH for macOS
	existingDYLDPath := os.Getenv("DYLD_LIBRARY_PATH")
	if existingDYLDPath != "" {
		cmd.Env = append(cmd.Env, fmt.Sprintf("DYLD_LIBRARY_PATH=%s:%s", candleLibPath, existingDYLDPath))
	} else {
		cmd.Env = append(cmd.Env, fmt.Sprintf("DYLD_LIBRARY_PATH=%s", candleLibPath))
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start router: %w", err)
	}

	// Store PID for later management (Issue #1: kill process if PID file write fails)
	pid := cmd.Process.Pid
	if err := os.WriteFile(pidFilePath, []byte(fmt.Sprintf("%d", pid)), 0o600); err != nil {
		// Kill process if we can't track it
		_ = cmd.Process.Kill()
		return fmt.Errorf("failed to write PID file: %w", err)
	}

	cli.Success(fmt.Sprintf("Router started (PID: %d)", pid))
	cli.Info(fmt.Sprintf("PID file: %s", pidFilePath))
	cli.Info(fmt.Sprintf("Log file: %s", logFilePath))
	cli.Info("To stop: vsr undeploy local")

	return nil // Don't wait, run in background
}

// DeployDocker deploys using Docker Compose
func DeployDocker(configPath string, withObservability bool) error {
	cli.Info("Deploying router with Docker Compose...")

	// Validate the configuration first
	cfg, err := config.Parse(configPath)
	if err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}
	if err := cli.ValidateConfig(cfg); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	// Check if docker-compose exists
	if !commandExists("docker-compose") && !commandExists("docker compose") {
		return fmt.Errorf("docker-compose not found. Please install Docker Compose")
	}

	// Download models first
	cli.Info("Downloading models...")
	cmd := exec.Command("make", "download-models")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to download models: %w", err)
	}

	// Determine compose file path
	composeFile := "deploy/docker-compose/docker-compose.yml"
	if _, err := os.Stat(composeFile); os.IsNotExist(err) {
		return fmt.Errorf("docker-compose file not found: %s", composeFile)
	}

	// Run docker-compose up
	var upCmd *exec.Cmd
	if commandExists("docker-compose") {
		upCmd = exec.Command("docker-compose", "-f", composeFile, "up", "-d")
	} else {
		upCmd = exec.Command("docker", "compose", "-f", composeFile, "up", "-d")
	}

	upCmd.Stdout = os.Stdout
	upCmd.Stderr = os.Stderr

	if err := upCmd.Run(); err != nil {
		return fmt.Errorf("failed to deploy with docker-compose: %w", err)
	}

	cli.Success("Router deployed with Docker Compose")
	cli.Info("Check status with: vsr status")
	cli.Info("View logs with: vsr logs")

	return nil
}

// DeployKubernetes deploys to Kubernetes
func DeployKubernetes(configPath, namespace string, withObservability bool) error {
	cli.Info("Deploying router to Kubernetes...")

	// Pre-deployment checks
	cli.Info("Running pre-deployment checks...")

	// 1. Check if kubectl exists
	if !commandExists("kubectl") {
		cli.Error("kubectl not found")
		cli.Info("Install kubectl: https://kubernetes.io/docs/tasks/tools/")
		return fmt.Errorf("kubectl not found")
	}

	// 2. Check cluster connectivity
	cli.Info("Checking cluster connectivity...")
	clusterInfoCmd := exec.Command("kubectl", "cluster-info")
	if err := clusterInfoCmd.Run(); err != nil {
		cli.Error("Unable to connect to Kubernetes cluster")
		cli.Info("Check your kubeconfig: kubectl config view")
		cli.Info("List available contexts: kubectl config get-contexts")
		return fmt.Errorf("no connection to Kubernetes cluster")
	}
	cli.Success("Cluster connection verified")

	// 3. Check/create namespace
	cli.Info(fmt.Sprintf("Checking namespace '%s'...", namespace))
	nsCheckCmd := exec.Command("kubectl", "get", "namespace", namespace)
	if err := nsCheckCmd.Run(); err != nil {
		// Namespace doesn't exist, create it
		cli.Info(fmt.Sprintf("Creating namespace '%s'...", namespace))
		nsCreateCmd := exec.Command("kubectl", "create", "namespace", namespace)
		nsCreateCmd.Stdout = os.Stdout
		nsCreateCmd.Stderr = os.Stderr
		if err := nsCreateCmd.Run(); err != nil {
			cli.Warning(fmt.Sprintf("Failed to create namespace: %v", err))
			cli.Info("You may need to create it manually: kubectl create namespace " + namespace)
		} else {
			cli.Success("Namespace created")
		}
	} else {
		cli.Success("Namespace exists")
	}

	// 4. Check permissions
	cli.Info("Checking permissions...")
	permCheckCmd := exec.Command("kubectl", "auth", "can-i", "create", "pods", "-n", namespace)
	if err := permCheckCmd.Run(); err != nil {
		cli.Warning("You may not have sufficient permissions")
		cli.Info("Check RBAC: kubectl auth can-i create pods -n " + namespace)
		cli.Info("You may need cluster-admin privileges for deployment")
	} else {
		cli.Success("Permissions verified")
	}

	// Apply manifests
	cli.Info("Applying Kubernetes manifests...")
	manifestDir := "deploy/kubernetes"
	if _, err := os.Stat(manifestDir); os.IsNotExist(err) {
		return fmt.Errorf("kubernetes manifests not found: %s", manifestDir)
	}

	cmd := exec.Command("kubectl", "apply", "-f", manifestDir, "-n", namespace)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to apply kubernetes manifests: %w", err)
	}

	cli.Success("Manifests applied successfully")

	// Post-deployment validation
	cli.Info("Waiting for pods to be ready...")
	timeout := 300 // 5 minutes
	ready := false

	for i := 0; i < timeout; i += 5 {
		time.Sleep(5 * time.Second)

		// Check pod status
		podsCmd := exec.Command("kubectl", "get", "pods", "-n", namespace, "-l", "app=semantic-router", "--no-headers")
		output, err := podsCmd.Output()
		if err != nil {
			cli.Info(fmt.Sprintf("Waiting for pods... (%ds/%ds)", i+5, timeout))
			continue
		}

		if len(output) == 0 {
			cli.Info(fmt.Sprintf("Waiting for pods to be created... (%ds/%ds)", i+5, timeout))
			continue
		}

		// Count ready pods
		lines := splitLines(string(output))
		totalPods := len(lines)
		readyPods := 0

		for _, line := range lines {
			if len(line) > 0 {
				// Simple check: if line contains "Running" and "1/1" or "2/2", etc.
				// This is a basic heuristic
				if containsString(line, "Running") {
					readyPods++
				}
			}
		}

		if readyPods > 0 && readyPods == totalPods {
			ready = true
			cli.Success(fmt.Sprintf("All %d pod(s) are ready", readyPods))
			break
		}

		if i%10 == 0 {
			cli.Info(fmt.Sprintf("Waiting for pods... (%d/%d ready, %ds/%ds)", readyPods, totalPods, i+5, timeout))
		}
	}

	if !ready {
		cli.Warning("Timeout waiting for pods to be ready")
		cli.Info("Check pod status: kubectl get pods -n " + namespace)
		cli.Info("Check pod logs: kubectl logs -n " + namespace + " -l app=semantic-router")
		return fmt.Errorf("pods did not become ready within timeout")
	}

	// Check deployment rollout status
	cli.Info("Checking deployment rollout status...")
	rolloutCmd := exec.Command("kubectl", "rollout", "status", "deployment/semantic-router", "-n", namespace, "--timeout=60s")
	rolloutCmd.Stdout = os.Stdout
	rolloutCmd.Stderr = os.Stderr
	if err := rolloutCmd.Run(); err != nil {
		cli.Warning("Deployment rollout check failed (pods may still be starting)")
	}

	// Verify service endpoints
	cli.Info("Verifying service endpoints...")
	svcCmd := exec.Command("kubectl", "get", "svc", "-n", namespace, "-l", "app=semantic-router")
	svcCmd.Stdout = os.Stdout
	svcCmd.Stderr = os.Stderr
	if err := svcCmd.Run(); err != nil {
		cli.Warning("Could not verify service endpoints")
	}

	cli.Success(fmt.Sprintf("Router deployed successfully to Kubernetes namespace: %s", namespace))
	cli.Info("\nNext steps:")
	cli.Info("  Check status: kubectl get pods -n " + namespace)
	cli.Info("  View logs: kubectl logs -n " + namespace + " -l app=semantic-router")
	cli.Info("  Port forward: kubectl port-forward -n " + namespace + " svc/semantic-router 8080:8080")

	return nil
}

// UndeployLocal stops the local router process
func UndeployLocal() error {
	cli.Info("Stopping local router...")

	// Get cross-platform file paths
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Check if PID file exists
	if _, err := os.Stat(pidFilePath); os.IsNotExist(err) {
		cli.Warning("No PID file found. Router may not be running.")
		cli.Info("Use: ps aux | grep router")
		return nil
	}

	// Read PID from file
	pidBytes, err := os.ReadFile(pidFilePath)
	if err != nil {
		return fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(string(pidBytes))
	if err != nil {
		return fmt.Errorf("invalid PID in file: %w", err)
	}

	// Find the process
	process, err := os.FindProcess(pid)
	if err != nil {
		cli.Warning(fmt.Sprintf("Process %d not found (may have already stopped)", pid))
		// Clean up PID file anyway
		os.Remove(pidFilePath)
		return nil
	}

	// Send SIGTERM for graceful shutdown
	cli.Info(fmt.Sprintf("Sending SIGTERM to process %d...", pid))
	if err := process.Signal(syscall.SIGTERM); err != nil {
		// Process might already be dead
		cli.Warning(fmt.Sprintf("Failed to send SIGTERM: %v", err))
	}

	// Wait for up to 10 seconds for graceful shutdown
	stopped := false
	for i := 0; i < 10; i++ {
		time.Sleep(1 * time.Second)
		// Try to signal with 0 to check if process exists
		if err := process.Signal(syscall.Signal(0)); err != nil {
			// Process is gone
			stopped = true
			break
		}
		cli.Info(fmt.Sprintf("Waiting for graceful shutdown... (%d/10s)", i+1))
	}

	// If still running, send SIGKILL
	if !stopped {
		cli.Warning("Process did not stop gracefully, sending SIGKILL...")
		if err := process.Kill(); err != nil {
			cli.Warning(fmt.Sprintf("Failed to kill process: %v", err))
		}
		time.Sleep(1 * time.Second)
	}

	// Clean up PID file
	if err := os.Remove(pidFilePath); err != nil {
		cli.Warning(fmt.Sprintf("Failed to remove PID file: %v", err))
	}

	// Optionally clean up log file (keep it for now for debugging)
	// os.Remove(logFilePath)

	cli.Success("Router stopped successfully")
	cli.Info(fmt.Sprintf("Log file available at: %s", logFilePath))
	return nil
}

// UndeployDocker removes Docker Compose deployment
func UndeployDocker(removeVolumes bool) error {
	cli.Info("Removing Docker Compose deployment...")

	composeFile := "deploy/docker-compose/docker-compose.yml"

	// Check if docker-compose file exists
	if _, err := os.Stat(composeFile); os.IsNotExist(err) {
		return fmt.Errorf("docker-compose file not found: %s", composeFile)
	}

	// Get list of containers before stopping
	cli.Info("Identifying running containers...")
	containersBefore, _ := getDockerContainers("semantic-router")

	// Build docker-compose down command
	var args []string
	if commandExists("docker-compose") {
		args = []string{"-f", composeFile, "down"}
	} else {
		args = []string{"compose", "-f", composeFile, "down"}
	}

	// Add --volumes flag if requested
	if removeVolumes {
		args = append(args, "--volumes")
		cli.Info("Will remove volumes...")
	}

	// Execute docker-compose down
	var cmd *exec.Cmd
	if commandExists("docker-compose") {
		cmd = exec.Command("docker-compose", args...)
	} else {
		cmd = exec.Command("docker", args...)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to undeploy: %w", err)
	}

	// Wait for containers to fully stop (max 30 seconds)
	cli.Info("Waiting for containers to stop...")
	stopped := false
	for i := 0; i < 30; i++ {
		time.Sleep(1 * time.Second)
		containers, _ := getDockerContainers("semantic-router")
		if len(containers) == 0 {
			stopped = true
			break
		}
		if i%5 == 4 { // Show progress every 5 seconds
			cli.Info(fmt.Sprintf("Still stopping... (%d/30s, %d containers remaining)", i+1, len(containers)))
		}
	}

	if !stopped {
		cli.Warning("Some containers may still be stopping")
	}

	// Verify cleanup
	containersAfter, _ := getDockerContainers("semantic-router")
	if len(containersAfter) > 0 {
		cli.Warning(fmt.Sprintf("Warning: %d container(s) still running", len(containersAfter)))
		for _, container := range containersAfter {
			cli.Warning(fmt.Sprintf("  - %s", container))
		}
	} else {
		cli.Success("All containers stopped successfully")
	}

	// Show cleanup summary
	if len(containersBefore) > 0 {
		cli.Info(fmt.Sprintf("Cleaned up %d container(s):", len(containersBefore)))
		for _, container := range containersBefore {
			cli.Info(fmt.Sprintf("  ✓ %s", container))
		}
	}

	if removeVolumes {
		cli.Success("Router undeployed (volumes removed)")
	} else {
		cli.Success("Router undeployed (volumes preserved)")
		cli.Info("To remove volumes, use: vsr undeploy docker --volumes")
	}

	return nil
}

// UndeployKubernetes removes Kubernetes deployment
func UndeployKubernetes(namespace string, wait bool) error {
	cli.Info("Removing Kubernetes deployment...")

	// Check if kubectl exists
	if !commandExists("kubectl") {
		return fmt.Errorf("kubectl not found. Please install kubectl")
	}

	// Check if namespace exists
	checkCmd := exec.Command("kubectl", "get", "namespace", namespace)
	if err := checkCmd.Run(); err != nil {
		cli.Warning(fmt.Sprintf("Namespace '%s' not found or not accessible", namespace))
		return nil
	}

	manifestDir := "deploy/kubernetes"
	if _, err := os.Stat(manifestDir); os.IsNotExist(err) {
		cli.Warning(fmt.Sprintf("Manifest directory not found: %s", manifestDir))
		cli.Info("Attempting to delete by label...")
		// Try deleting by common labels
		labelCmd := exec.Command("kubectl", "delete", "all", "-l", "app=semantic-router", "-n", namespace)
		labelCmd.Stdout = os.Stdout
		labelCmd.Stderr = os.Stderr
		if err := labelCmd.Run(); err != nil {
			return fmt.Errorf("failed to delete resources: %w", err)
		}
	} else {
		// Delete using manifest files
		cmd := exec.Command("kubectl", "delete", "-f", manifestDir, "-n", namespace)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			cli.Warning(fmt.Sprintf("Some resources may not have been deleted: %v", err))
			// Don't return error, continue to wait/verify
		}
	}

	// Wait for pods to terminate if requested
	if wait {
		cli.Info("Waiting for pods to terminate...")
		timeout := 5 * 60 // 5 minutes in seconds
		stopped := false

		for i := 0; i < timeout; i += 2 {
			time.Sleep(2 * time.Second)

			// Check for pods
			checkCmd := exec.Command("kubectl", "get", "pods", "-n", namespace, "-l", "app=semantic-router", "--no-headers")
			output, err := checkCmd.Output()

			if err != nil || len(output) == 0 {
				// No pods found or error (likely no resources)
				stopped = true
				break
			}

			// Count remaining pods
			podCount := len(splitLines(string(output)))
			if podCount == 0 {
				stopped = true
				break
			}

			// Show progress every 10 seconds
			if i%10 == 0 {
				cli.Info(fmt.Sprintf("Waiting for pods to terminate... (%ds/%ds, %d pods remaining)", i, timeout, podCount))
			}
		}

		if !stopped {
			cli.Warning("Timeout waiting for all pods to terminate")
			cli.Info("Some resources may still be terminating in the background")
		} else {
			cli.Success("All pods terminated successfully")
		}
	}

	// Verify cleanup
	verifyCmd := exec.Command("kubectl", "get", "all", "-n", namespace, "-l", "app=semantic-router", "--no-headers")
	output, err := verifyCmd.Output()
	if err == nil && len(output) > 0 {
		remainingResources := len(splitLines(string(output)))
		if remainingResources > 0 {
			cli.Warning(fmt.Sprintf("Warning: %d resource(s) may still exist", remainingResources))
			cli.Info("Check with: kubectl get all -n " + namespace + " -l app=semantic-router")
		}
	}

	cli.Success(fmt.Sprintf("Router undeployed from Kubernetes namespace: %s", namespace))
	if !wait {
		cli.Info("Resources may still be terminating in the background")
		cli.Info("Use --wait flag to wait for complete cleanup")
	}

	return nil
}

// CheckStatus checks the status of all deployments
func CheckStatus(namespace string) error {
	cli.Info("Checking router status...")

	foundAny := false

	// Check local deployment
	localStatus := DetectLocalDeployment()
	if localStatus.IsRunning {
		foundAny = true
		displayDeploymentStatus(localStatus)
	}

	// Check Docker deployment
	dockerStatus := DetectDockerDeployment()
	if dockerStatus.IsRunning {
		foundAny = true
		displayDeploymentStatus(dockerStatus)
	}

	// Check Kubernetes deployment
	k8sStatus := DetectKubernetesDeployment(namespace)
	if k8sStatus.IsRunning {
		foundAny = true
		displayDeploymentStatus(k8sStatus)
	}

	// Check Helm deployment
	helmStatus := DetectHelmDeployment(namespace)
	if helmStatus.IsRunning {
		foundAny = true
		displayDeploymentStatus(helmStatus)
	}

	if !foundAny {
		cli.Warning("No router deployments found")
		cli.Info("Deploy the router with: vsr deploy [local|docker|kubernetes|helm]")
	}

	return nil
}

// DetectLocalDeployment checks for local deployment
func DetectLocalDeployment() *DeploymentStatus {
	status := &DeploymentStatus{
		Type:      "local",
		IsRunning: false,
	}

	// Get cross-platform file paths
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Check if PID file exists
	if _, err := os.Stat(pidFilePath); err == nil {
		pidBytes, err := os.ReadFile(pidFilePath)
		if err == nil {
			pid, err := strconv.Atoi(string(pidBytes))
			if err == nil {
				// Check if process is running
				process, err := os.FindProcess(pid)
				if err == nil {
					// Try to signal the process
					if err := process.Signal(syscall.Signal(0)); err == nil {
						status.IsRunning = true
						status.Components = []ComponentStatus{
							{
								Name:    "router",
								Status:  "running",
								Message: fmt.Sprintf("PID: %d", pid),
							},
						}
						status.Endpoints = []string{
							"Check logs: " + logFilePath,
						}
					}
				}
			}
		}
	}

	return status
}

// DetectDockerDeployment checks for Docker Compose deployment
func DetectDockerDeployment() *DeploymentStatus {
	status := &DeploymentStatus{
		Type:      "docker-compose",
		IsRunning: false,
	}

	if !isDockerRunning() {
		return status
	}

	// Get Docker containers
	containers, err := getDockerContainers("semantic-router")
	if err != nil || len(containers) == 0 {
		return status
	}

	status.IsRunning = true
	status.Components = []ComponentStatus{}
	status.Endpoints = []string{
		"Router API: http://localhost:8080",
		"Envoy Proxy: http://localhost:8801",
		"Dashboard: http://localhost:8700",
		"Grafana: http://localhost:3000",
	}

	// Get detailed status for each container
	for _, container := range containers {
		inspectCmd := exec.Command("docker", "inspect", "--format", "{{.State.Status}}", container)
		output, err := inspectCmd.Output()
		containerStatus := "unknown"
		if err == nil {
			containerStatus = strings.TrimSpace(string(output))
		}

		status.Components = append(status.Components, ComponentStatus{
			Name:    container,
			Status:  containerStatus,
			Message: "",
		})
	}

	return status
}

// DetectKubernetesDeployment checks for Kubernetes deployment
func DetectKubernetesDeployment(namespace string) *DeploymentStatus {
	status := &DeploymentStatus{
		Type:      "kubernetes",
		IsRunning: false,
		Namespace: namespace,
	}

	if !commandExists("kubectl") {
		return status
	}

	// Check for pods
	cmd := exec.Command("kubectl", "get", "pods", "-n", namespace, "-l", "app=semantic-router", "--no-headers")
	output, err := cmd.Output()
	if err != nil || len(output) == 0 {
		return status
	}

	lines := splitLines(string(output))
	if len(lines) == 0 {
		return status
	}

	status.IsRunning = true
	status.Components = []ComponentStatus{}

	for _, line := range lines {
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 3 {
			podName := fields[0]
			podStatus := fields[2]
			status.Components = append(status.Components, ComponentStatus{
				Name:    podName,
				Status:  podStatus,
				Message: "",
			})
		}
	}

	// Get service info
	svcCmd := exec.Command("kubectl", "get", "svc", "-n", namespace, "-l", "app=semantic-router", "--no-headers")
	svcOutput, err := svcCmd.Output()
	if err == nil && len(svcOutput) > 0 {
		status.Endpoints = []string{
			fmt.Sprintf("Check services: kubectl get svc -n %s", namespace),
		}
	}

	return status
}

// displayDeploymentStatus displays the status of a deployment
func displayDeploymentStatus(status *DeploymentStatus) {
	cli.Info("\n╔═══════════════════════════════════════╗")
	cli.Info(fmt.Sprintf("║ Deployment: %-26s║", status.Type))
	cli.Info("╚═══════════════════════════════════════╝")

	if status.Namespace != "" {
		cli.Info(fmt.Sprintf("Namespace: %s", status.Namespace))
	}

	if status.ReleaseName != "" {
		cli.Info(fmt.Sprintf("Release: %s", status.ReleaseName))
	}

	// Show components
	if len(status.Components) > 0 {
		cli.Info("\nComponents:")
		for _, comp := range status.Components {
			statusSymbol := "✓"
			if comp.Status != "running" && comp.Status != "Running" {
				statusSymbol = "⚠"
			}
			msg := comp.Message
			if msg != "" {
				cli.Info(fmt.Sprintf("  %s %-30s %-15s %s", statusSymbol, comp.Name, comp.Status, msg))
			} else {
				cli.Info(fmt.Sprintf("  %s %-30s %s", statusSymbol, comp.Name, comp.Status))
			}
		}
	}

	// Show endpoints
	if len(status.Endpoints) > 0 {
		cli.Info("\nEndpoints:")
		for _, endpoint := range status.Endpoints {
			cli.Info(fmt.Sprintf("  %s", endpoint))
		}
	}

	fmt.Println() // Extra newline for spacing
}

// FetchLogs fetches logs from the router with auto-detection
func FetchLogs(follow bool, tail int, namespace, deployType, component string, since string, grep string) error {
	cli.Info("Fetching router logs...")

	// Auto-detect deployment type if not specified
	if deployType == "" {
		deployType = detectDeploymentType(namespace)
		if deployType == "" {
			cli.Warning("Could not detect router deployment")
			cli.Info("Specify deployment type with: vsr logs --env [local|docker|kubernetes|helm]")
			return fmt.Errorf("no router deployment found")
		}
		cli.Info(fmt.Sprintf("Detected deployment type: %s", deployType))
	}

	// Fetch logs based on deployment type
	switch deployType {
	case "local":
		return fetchLocalLogs(follow, tail, since, grep)
	case "docker":
		return fetchDockerLogsEnhanced(follow, tail, component, since, grep)
	case "kubernetes":
		return fetchKubernetesLogs(follow, tail, namespace, component, since, grep)
	case "helm":
		return fetchHelmLogs(follow, tail, namespace, component, since, grep)
	default:
		return fmt.Errorf("unsupported deployment type: %s", deployType)
	}
}

// detectDeploymentType detects the deployment type
func detectDeploymentType(namespace string) string {
	// Check in order of specificity
	if DetectHelmDeployment(namespace).IsRunning {
		return "helm"
	}
	if DetectKubernetesDeployment(namespace).IsRunning {
		return "kubernetes"
	}
	if DetectDockerDeployment().IsRunning {
		return "docker"
	}
	if DetectLocalDeployment().IsRunning {
		return "local"
	}
	return ""
}

// fetchLocalLogs fetches logs from local deployment
func fetchLocalLogs(follow bool, tail int, since string, grep string) error {
	logFilePath := getLogFilePath()

	if _, err := os.Stat(logFilePath); os.IsNotExist(err) {
		return fmt.Errorf("log file not found: %s", logFilePath)
	}

	if follow {
		// Use tail -f for following logs
		args := []string{"-f"}
		if tail > 0 {
			args = append(args, "-n", fmt.Sprintf("%d", tail))
		}
		args = append(args, logFilePath)

		cmd := exec.Command("tail", args...)

		if grep != "" {
			// Pipe through grep if pattern specified
			grepCmd := exec.Command("grep", "--color=always", grep)
			grepCmd.Stdin, _ = cmd.StdoutPipe()
			grepCmd.Stdout = os.Stdout
			grepCmd.Stderr = os.Stderr

			if err := cmd.Start(); err != nil {
				return fmt.Errorf("failed to start tail: %w", err)
			}
			if err := grepCmd.Run(); err != nil {
				_ = cmd.Process.Kill()
				return fmt.Errorf("grep failed: %w", err)
			}
			return cmd.Wait()
		}

		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}

	// Non-following mode - just cat with tail
	args := []string{}
	if tail > 0 {
		args = append(args, "-n", fmt.Sprintf("%d", tail))
	}
	args = append(args, logFilePath)

	cmd := exec.Command("tail", args...)

	if grep != "" {
		grepCmd := exec.Command("grep", "--color=always", grep)
		grepCmd.Stdin, _ = cmd.StdoutPipe()
		grepCmd.Stdout = os.Stdout
		grepCmd.Stderr = os.Stderr

		if err := cmd.Start(); err != nil {
			return fmt.Errorf("failed to start tail: %w", err)
		}
		if err := grepCmd.Run(); err != nil {
			_ = cmd.Process.Kill()
			return fmt.Errorf("grep failed: %w", err)
		}
		return cmd.Wait()
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// fetchDockerLogsEnhanced fetches logs from Docker Compose deployment
func fetchDockerLogsEnhanced(follow bool, tail int, component string, since string, grep string) error {
	if !isDockerRunning() {
		return fmt.Errorf("docker is not running")
	}

	// Get list of containers
	containers, err := getDockerContainers("semantic-router")
	if err != nil || len(containers) == 0 {
		return fmt.Errorf("no semantic-router containers found")
	}

	// Filter by component if specified
	targetContainers := containers
	if component != "" && component != "all" {
		targetContainers = []string{}
		for _, container := range containers {
			if containsString(container, component) {
				targetContainers = append(targetContainers, container)
			}
		}
		if len(targetContainers) == 0 {
			return fmt.Errorf("no containers found matching component: %s", component)
		}
	}

	// Build docker logs command
	for _, container := range targetContainers {
		cli.Info(fmt.Sprintf("=== Logs from: %s ===", container))

		args := []string{"logs"}
		if follow {
			args = append(args, "-f")
		}
		if tail > 0 {
			args = append(args, "--tail", fmt.Sprintf("%d", tail))
		}
		if since != "" {
			args = append(args, "--since", since)
		}
		args = append(args, container)

		cmd := exec.Command("docker", args...)

		if grep != "" {
			grepCmd := exec.Command("grep", "--color=always", grep)
			grepCmd.Stdin, _ = cmd.StdoutPipe()
			grepCmd.Stdout = os.Stdout
			grepCmd.Stderr = os.Stderr

			if err := cmd.Start(); err != nil {
				return fmt.Errorf("failed to start docker logs: %w", err)
			}
			if err := grepCmd.Run(); err != nil {
				_ = cmd.Process.Kill()
				return fmt.Errorf("grep failed: %w", err)
			}
			if err := cmd.Wait(); err != nil {
				return err
			}
		} else {
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to fetch logs from %s: %w", container, err)
			}
		}

		fmt.Println() // Add spacing between containers
	}

	return nil
}

// fetchKubernetesLogs fetches logs from Kubernetes deployment
func fetchKubernetesLogs(follow bool, tail int, namespace string, component string, since string, grep string) error {
	if !commandExists("kubectl") {
		return fmt.Errorf("kubectl not found")
	}

	// Build label selector
	labelSelector := "app=semantic-router"
	if component != "" && component != "all" {
		labelSelector = fmt.Sprintf("app=semantic-router,component=%s", component)
	}

	// Build kubectl logs command
	args := []string{"logs", "-n", namespace, "-l", labelSelector}
	if follow {
		args = append(args, "-f")
	}
	if tail > 0 {
		args = append(args, "--tail", fmt.Sprintf("%d", tail))
	}
	if since != "" {
		args = append(args, "--since", since)
	}
	args = append(args, "--all-containers=true", "--prefix=true")

	cmd := exec.Command("kubectl", args...)

	if grep != "" {
		grepCmd := exec.Command("grep", "--color=always", grep)
		grepCmd.Stdin, _ = cmd.StdoutPipe()
		grepCmd.Stdout = os.Stdout
		grepCmd.Stderr = os.Stderr

		if err := cmd.Start(); err != nil {
			return fmt.Errorf("failed to start kubectl logs: %w", err)
		}
		if err := grepCmd.Run(); err != nil {
			_ = cmd.Process.Kill()
			return fmt.Errorf("grep failed: %w", err)
		}
		return cmd.Wait()
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// fetchHelmLogs fetches logs from Helm deployment
func fetchHelmLogs(follow bool, tail int, namespace string, component string, since string, grep string) error {
	// Helm deployments use Kubernetes, so we can reuse the K8s log fetching
	// but with different label selector
	if !commandExists("kubectl") {
		return fmt.Errorf("kubectl not found")
	}

	// Get release name
	helmStatus := DetectHelmDeployment(namespace)
	if !helmStatus.IsRunning {
		return fmt.Errorf("no helm deployment found in namespace: %s", namespace)
	}

	// Build label selector for Helm
	labelSelector := fmt.Sprintf("app.kubernetes.io/instance=%s", helmStatus.ReleaseName)
	if component != "" && component != "all" {
		labelSelector = fmt.Sprintf("%s,app.kubernetes.io/component=%s", labelSelector, component)
	}

	// Build kubectl logs command
	args := []string{"logs", "-n", namespace, "-l", labelSelector}
	if follow {
		args = append(args, "-f")
	}
	if tail > 0 {
		args = append(args, "--tail", fmt.Sprintf("%d", tail))
	}
	if since != "" {
		args = append(args, "--since", since)
	}
	args = append(args, "--all-containers=true", "--prefix=true")

	cmd := exec.Command("kubectl", args...)

	if grep != "" {
		grepCmd := exec.Command("grep", "--color=always", grep)
		grepCmd.Stdin, _ = cmd.StdoutPipe()
		grepCmd.Stdout = os.Stdout
		grepCmd.Stderr = os.Stderr

		if err := cmd.Start(); err != nil {
			return fmt.Errorf("failed to start kubectl logs: %w", err)
		}
		if err := grepCmd.Run(); err != nil {
			_ = cmd.Process.Kill()
			return fmt.Errorf("grep failed: %w", err)
		}
		return cmd.Wait()
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// Helper functions

func buildRouter() error {
	cmd := exec.Command("make", "build")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func commandExists(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

func isDockerRunning() bool {
	cmd := exec.Command("docker", "ps")
	return cmd.Run() == nil
}

func getDockerContainers(nameFilter string) ([]string, error) {
	//nolint:gosec // G204: nameFilter is from internal use, not user input
	cmd := exec.Command("docker", "ps", "--filter", fmt.Sprintf("name=%s", nameFilter), "--format", "{{.Names}}")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	containers := []string{}
	if len(output) > 0 {
		lines := string(output)
		for _, line := range splitLines(lines) {
			if line != "" {
				containers = append(containers, line)
			}
		}
	}
	return containers, nil
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i, c := range s {
		if c == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
