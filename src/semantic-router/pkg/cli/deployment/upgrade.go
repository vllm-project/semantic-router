package deployment

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/timeout"
)

// UpgradeLocal upgrades the local router deployment
func UpgradeLocal(configPath string) error {
	cli.Info("Upgrading local router...")

	// Get cross-platform file path
	pidFilePath := getPIDFilePath()

	// Check if PID file exists (router is running)
	if _, err := os.Stat(pidFilePath); os.IsNotExist(err) {
		cli.Warning("No running local router found")
		cli.Info("Use 'vsr deploy local' to start a new deployment")
		return nil
	}

	// Read current PID
	pidBytes, err := os.ReadFile(pidFilePath)
	if err != nil {
		return fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(string(pidBytes))
	if err != nil {
		return fmt.Errorf("invalid PID in file: %w", err)
	}

	cli.Info(fmt.Sprintf("Found running router (PID: %d)", pid))

	// Rebuild the router binary
	cli.Info("Rebuilding router binary...")
	if buildErr := buildRouter(); buildErr != nil {
		return fmt.Errorf("failed to rebuild router: %w", buildErr)
	}
	cli.Success("Binary rebuilt successfully")

	// Find the process
	process, err := os.FindProcess(pid)
	if err != nil {
		cli.Warning(fmt.Sprintf("Process %d not found", pid))
		// Try to deploy fresh
		return DeployLocal(configPath, false)
	}

	// Send SIGTERM for graceful shutdown
	cli.Info("Stopping current router...")
	if err := process.Signal(syscall.SIGTERM); err != nil {
		cli.Warning(fmt.Sprintf("Failed to send SIGTERM: %v", err))
	}

	// Wait for process to stop (max 10 seconds)
	stopped := false
	for i := 0; i < 10; i++ {
		time.Sleep(1 * time.Second)
		if err := process.Signal(syscall.Signal(0)); err != nil {
			stopped = true
			break
		}
	}

	// Force kill if needed
	if !stopped {
		cli.Warning("Forcing process termination...")
		_ = process.Kill()
		time.Sleep(1 * time.Second)
	}

	// Clean up old PID file
	os.Remove(pidFilePath)

	cli.Success("Old router stopped")

	// Start new version
	cli.Info("Starting upgraded router...")
	if err := DeployLocal(configPath, false); err != nil {
		return fmt.Errorf("failed to start upgraded router: %w", err)
	}

	cli.Success("Local router upgraded successfully")
	return nil
}

// UpgradeDocker upgrades the Docker Compose deployment
func UpgradeDocker(configPath string, withObservability bool) error {
	cli.Info("Upgrading Docker deployment...")

	// Check if docker-compose is running
	if !isDockerRunning() {
		cli.Warning("No running Docker deployment found")
		cli.Info("Use 'vsr deploy docker' to start a new deployment")
		return nil
	}

	composeFile := "deploy/docker-compose/docker-compose.yml"
	if _, err := os.Stat(composeFile); os.IsNotExist(err) {
		return fmt.Errorf("docker-compose file not found: %s", composeFile)
	}

	// Pull latest images
	cli.Info("Pulling latest Docker images...")
	var pullCmd *exec.Cmd
	if commandExists("docker-compose") {
		pullCmd = exec.Command("docker-compose", "-f", composeFile, "pull")
	} else {
		pullCmd = exec.Command("docker", "compose", "-f", composeFile, "pull")
	}
	if err := timeout.RunCommandWithIdleTimeoutContext(
		context.Background(),
		pullCmd,
		timeout.DefaultConfig,
		"docker-compose pull",
	); err != nil {
		if timeout.IsIdleTimeout(err) {
			return fmt.Errorf("image pull failed: %w\n"+
				"Troubleshooting:\n"+
				"  - Check internet connection\n"+
				"  - Verify Docker Hub access: docker login\n"+
				"  - Check disk space: df -h",
				err)
		}
		return fmt.Errorf("failed to pull latest images: %w", err)
	}
	cli.Success("Images pulled successfully")

	// Recreate containers with new images
	cli.Info("Recreating containers...")
	var upCmd *exec.Cmd
	if commandExists("docker-compose") {
		upCmd = exec.Command("docker-compose", "-f", composeFile, "up", "-d", "--force-recreate", "--no-deps")
	} else {
		upCmd = exec.Command("docker", "compose", "-f", composeFile, "up", "-d", "--force-recreate", "--no-deps")
	}
	if err := timeout.RunCommandWithIdleTimeoutContext(
		context.Background(),
		upCmd,
		timeout.DefaultConfig,
		"docker-compose up",
	); err != nil {
		if timeout.IsIdleTimeout(err) {
			return fmt.Errorf("docker compose startup failed: %w\n"+
				"Troubleshooting:\n"+
				"  - Check Docker daemon: docker ps\n"+
				"  - Check logs: docker-compose logs\n"+
				"  - Verify ports: netstat -tulpn | grep LISTEN",
				err)
		}
		return fmt.Errorf("failed to recreate containers: %w", err)
	}

	// Wait for containers to be healthy
	cli.Info("Waiting for containers to be ready...")
	time.Sleep(5 * time.Second) // Give containers time to start

	// Check container health
	healthy := false
	for i := 0; i < 30; i++ {
		containers, _ := getDockerContainers("semantic-router")
		if len(containers) > 0 {
			// Simple health check - containers are running
			healthy = true
			break
		}
		time.Sleep(2 * time.Second)
		if i%5 == 0 {
			cli.Info(fmt.Sprintf("Waiting for containers... (%ds/60s)", i*2))
		}
	}

	if !healthy {
		cli.Warning("Could not verify container health")
		cli.Info("Check status with: vsr status")
		return fmt.Errorf("containers may not be healthy")
	}

	cli.Success("Docker deployment upgraded successfully")
	cli.Info("Check status with: vsr status")
	cli.Info("View logs with: vsr logs")
	return nil
}

// UpgradeKubernetes upgrades the Kubernetes deployment
func UpgradeKubernetes(configPath, namespace string, timeout int, wait bool) error {
	cli.Info("Upgrading Kubernetes deployment...")

	// Check if kubectl exists
	if !commandExists("kubectl") {
		return fmt.Errorf("kubectl not found. Please install kubectl")
	}

	// Check if deployment exists
	var checkErr error
	func() {
		// 10 second timeout for checking deployment existence
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		checkCmd := exec.CommandContext(ctx, "kubectl", "get", "deployment", "semantic-router", "-n", namespace)
		checkErr = checkCmd.Run()
	}()
	if checkErr != nil {
		if errors.Is(checkErr, context.DeadlineExceeded) {
			return fmt.Errorf("kubectl get deployment timed out after 10 seconds: %w", checkErr)
		}
		cli.Warning("No deployment found in namespace: " + namespace)
		cli.Info("Use 'vsr deploy kubernetes' to create a new deployment")
		return nil
	}

	// Apply updated manifests
	cli.Info("Applying updated manifests...")
	manifestDir := "deploy/kubernetes"
	if _, err := os.Stat(manifestDir); os.IsNotExist(err) {
		return fmt.Errorf("kubernetes manifests not found: %s", manifestDir)
	}

	var applyErr error
	func() {
		// 120 second timeout for applying manifests
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()
		applyCmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", manifestDir, "-n", namespace)
		applyCmd.Stdout = os.Stdout
		applyCmd.Stderr = os.Stderr
		applyErr = applyCmd.Run()
	}()

	if applyErr != nil {
		if errors.Is(applyErr, context.DeadlineExceeded) {
			return fmt.Errorf("kubectl apply timed out after 120 seconds: %w", applyErr)
		}
		return fmt.Errorf("failed to apply manifests: %w", applyErr)
	}
	cli.Success("Manifests applied successfully")

	// Trigger rolling restart
	cli.Info("Triggering rolling restart...")
	var restartErr error
	func() {
		// 10 second timeout for triggering restart
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		restartCmd := exec.CommandContext(ctx, "kubectl", "rollout", "restart", "deployment/semantic-router", "-n", namespace)
		restartCmd.Stdout = os.Stdout
		restartCmd.Stderr = os.Stderr
		restartErr = restartCmd.Run()
	}()

	if restartErr != nil {
		if errors.Is(restartErr, context.DeadlineExceeded) {
			return fmt.Errorf("kubectl rollout restart timed out after 10 seconds: %w", restartErr)
		}
		return fmt.Errorf("failed to restart deployment: %w", restartErr)
	}

	// Wait for rollout to complete if requested
	if wait {
		cli.Info("Waiting for rollout to complete...")
		//nolint:gosec // G204: namespace is from internal config
		rolloutCmd := exec.Command("kubectl", "rollout", "status", "deployment/semantic-router", "-n", namespace, fmt.Sprintf("--timeout=%ds", timeout))
		rolloutCmd.Stdout = os.Stdout
		rolloutCmd.Stderr = os.Stderr

		if err := rolloutCmd.Run(); err != nil {
			cli.Warning("Rollout status check failed")
			cli.Info("Check manually: kubectl rollout status deployment/semantic-router -n " + namespace)
			return fmt.Errorf("rollout may not have completed successfully: %w", err)
		}
		cli.Success("Rollout completed successfully")
	} else {
		cli.Info("Rollout started (not waiting for completion)")
		cli.Info("Monitor with: kubectl rollout status deployment/semantic-router -n " + namespace)
	}

	cli.Success("Kubernetes deployment upgraded successfully")
	cli.Info("Check status with: kubectl get pods -n " + namespace)
	return nil
}

// UpgradeHelm upgrades the Helm deployment
func UpgradeHelm(configPath, namespace string, timeout int) error {
	cli.Info("Upgrading Helm deployment...")

	// Check if helm exists
	if !commandExists("helm") {
		return fmt.Errorf("helm not found. Please install Helm: https://helm.sh/docs/intro/install/")
	}

	// Check if release exists
	var output []byte
	var err error
	func() {
		// 10 second timeout for listing helm releases
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		checkCmd := exec.CommandContext(ctx, "helm", "list", "-n", namespace, "-q")
		output, err = checkCmd.Output()
	}()
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("helm list timed out after 10 seconds: %w", err)
		}
		cli.Warning("No Helm release found in namespace: " + namespace)
		cli.Info("Use 'vsr deploy helm' to create a new deployment")
		return nil
	}
	if len(output) == 0 {
		cli.Warning("No Helm release found in namespace: " + namespace)
		cli.Info("Use 'vsr deploy helm' to create a new deployment")
		return nil
	}

	cli.Warning("Helm deployment upgrade is not fully implemented yet")
	cli.Info("This feature will be available in a future release")
	cli.Info("\nWorkaround:")
	cli.Info("1. Update your values.yaml file")
	cli.Info("2. Run: helm upgrade semantic-router ./deploy/helm/semantic-router -n " + namespace)
	cli.Info("3. Wait: kubectl rollout status deployment/semantic-router -n " + namespace)

	return fmt.Errorf("helm upgrade not yet implemented")
}
