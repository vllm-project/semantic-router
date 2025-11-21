package deployment

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// DeployLocal deploys the router as a local process
func DeployLocal(configPath string) error {
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

	// Start router process
	cmd := exec.Command(binPath, "--config", absConfigPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start router: %w", err)
	}

	cli.Success(fmt.Sprintf("Router started (PID: %d)", cmd.Process.Pid))
	cli.Info("To stop: kill " + fmt.Sprintf("%d", cmd.Process.Pid))

	return nil // Don't wait, run in background
}

// DeployDocker deploys using Docker Compose
func DeployDocker(configPath string, withObservability bool) error {
	cli.Info("Deploying router with Docker Compose...")

	// Check if docker-compose exists
	if !commandExists("docker-compose") && !commandExists("docker compose") {
		return fmt.Errorf("docker-compose not found. Please install Docker Compose")
	}

	// Determine compose file path
	composeFile := "deploy/docker-compose/docker-compose.yml"
	if _, err := os.Stat(composeFile); os.IsNotExist(err) {
		return fmt.Errorf("docker-compose file not found: %s", composeFile)
	}

	// Run docker-compose up
	var cmd *exec.Cmd
	if commandExists("docker-compose") {
		cmd = exec.Command("docker-compose", "-f", composeFile, "up", "-d")
	} else {
		cmd = exec.Command("docker", "compose", "-f", composeFile, "up", "-d")
	}

	// Capture stderr for error classification
	var stderr bytes.Buffer
	cmd.Stdout = os.Stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		errMsg := stderr.String()
		friendlyMsg := classifyDockerError(errMsg)

		if friendlyMsg != "" {
			cli.Error(fmt.Sprintf("Deployment failed: %s", friendlyMsg))
			cli.Info("Details: " + strings.TrimSpace(errMsg))
			return fmt.Errorf("docker deployment failed")
		}

		// If no classification matched, print raw error
		fmt.Fprint(os.Stderr, errMsg)
		return fmt.Errorf("failed to deploy with docker-compose: %w", err)
	}

	cli.Success("Router deployed with Docker Compose")
	cli.Info("Check status with: vsr status")
	cli.Info("View logs with: vsr logs")

	return nil
}

func classifyDockerError(errMsg string) string {
	errMsg = strings.ToLower(errMsg)

	if strings.Contains(errMsg, "error during connect") ||
		strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "daemon is not running") ||
		strings.Contains(errMsg, "dockerdesktoplinuxengine") {
		return "Docker Engine is not running or not reachable.\n   Please ensure Docker Desktop or the Docker daemon is started."
	}

	if strings.Contains(errMsg, "permission denied") {
		return "Permission denied when accessing Docker.\n   Please ensure you have permissions to run Docker (try 'sudo' or add user to 'docker' group)."
	}

	if strings.Contains(errMsg, "no such image") || strings.Contains(errMsg, "pull access denied") {
		return "Failed to pull required images.\n   Please check your internet connection and ensure you have access to the required repositories."
	}

	if strings.Contains(errMsg, "port is already allocated") || strings.Contains(errMsg, "address already in use") {
		return "Port conflict detected.\n   Please check if another service is using port 8080 or other required ports."
	}

	return ""
}

// DeployKubernetes deploys to Kubernetes
func DeployKubernetes(configPath, namespace string, withObservability bool) error {
	cli.Info("Deploying router to Kubernetes...")

	// Check if kubectl exists
	if !commandExists("kubectl") {
		return fmt.Errorf("kubectl not found. Please install kubectl")
	}

	// Apply manifests
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

	cli.Success(fmt.Sprintf("Router deployed to Kubernetes namespace: %s", namespace))
	cli.Info("Check status with: kubectl get pods -n " + namespace)

	return nil
}

// UndeployLocal stops the local router process
func UndeployLocal() error {
	cli.Warning("To stop local router, kill the process manually")
	cli.Info("Use: ps aux | grep router")
	return nil
}

// UndeployDocker removes Docker Compose deployment
func UndeployDocker() error {
	cli.Info("Removing Docker Compose deployment...")

	composeFile := "deploy/docker-compose/docker-compose.yml"

	var cmd *exec.Cmd
	if commandExists("docker-compose") {
		cmd = exec.Command("docker-compose", "-f", composeFile, "down")
	} else {
		cmd = exec.Command("docker", "compose", "-f", composeFile, "down")
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to undeploy: %w", err)
	}

	cli.Success("Router undeployed")
	return nil
}

// UndeployKubernetes removes Kubernetes deployment
func UndeployKubernetes(namespace string) error {
	cli.Info("Removing Kubernetes deployment...")

	manifestDir := "deploy/kubernetes"
	cmd := exec.Command("kubectl", "delete", "-f", manifestDir, "-n", namespace)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete kubernetes resources: %w", err)
	}

	cli.Success("Router undeployed from Kubernetes")
	return nil
}

// CheckStatus checks the status of the router
func CheckStatus() error {
	cli.Info("Checking router status...")

	// Try to detect deployment type and check status
	if isDockerRunning() {
		return checkDockerStatus()
	}

	cli.Warning("Could not detect router deployment")
	cli.Info("Deploy the router with: vsr deploy [local|docker|kubernetes]")
	return nil
}

// FetchLogs fetches logs from the router
func FetchLogs(follow bool, tail int) error {
	cli.Info("Fetching router logs...")

	if isDockerRunning() {
		return fetchDockerLogs(follow, tail)
	}

	cli.Warning("Could not detect router deployment")
	return nil
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

func checkDockerStatus() error {
	cmd := exec.Command("docker", "ps", "--filter", "name=semantic-router", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func fetchDockerLogs(follow bool, tail int) error {
	args := []string{"logs"}
	if follow {
		args = append(args, "-f")
	}
	args = append(args, "--tail", fmt.Sprintf("%d", tail), "semantic-router")

	cmd := exec.Command("docker", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
