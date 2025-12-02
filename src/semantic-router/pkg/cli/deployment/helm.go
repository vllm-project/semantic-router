package deployment

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

const (
	defaultHelmRelease = "semantic-router"
	defaultHelmChart   = "deploy/helm/semantic-router"
)

// DeployHelm deploys using Helm chart
func DeployHelm(configPath, namespace string, releaseName string, withObs bool, setValues []string) error {
	cli.Info("Deploying router with Helm...")

	// Pre-deployment checks
	cli.Info("Running pre-deployment checks...")

	// 1. Check if helm exists
	if !commandExists("helm") {
		cli.Error("helm not found")
		cli.Info("Install Helm: https://helm.sh/docs/intro/install/")
		return fmt.Errorf("helm not found")
	}

	// 2. Check if kubectl exists (Helm needs it)
	if !commandExists("kubectl") {
		cli.Error("kubectl not found")
		cli.Info("Install kubectl: https://kubernetes.io/docs/tasks/tools/")
		return fmt.Errorf("kubectl not found")
	}

	// 3. Check cluster connectivity
	cli.Info("Checking cluster connectivity...")
	clusterInfoCmd := exec.Command("kubectl", "cluster-info")
	if err := clusterInfoCmd.Run(); err != nil {
		cli.Error("Unable to connect to Kubernetes cluster")
		cli.Info("Check your kubeconfig: kubectl config view")
		return fmt.Errorf("no connection to Kubernetes cluster")
	}
	cli.Success("Cluster connection verified")

	// 4. Check/create namespace
	cli.Info(fmt.Sprintf("Checking namespace '%s'...", namespace))
	nsCheckCmd := exec.Command("kubectl", "get", "namespace", namespace)
	if err := nsCheckCmd.Run(); err != nil {
		cli.Info(fmt.Sprintf("Creating namespace '%s'...", namespace))
		nsCreateCmd := exec.Command("kubectl", "create", "namespace", namespace)
		if err := nsCreateCmd.Run(); err != nil {
			cli.Warning(fmt.Sprintf("Failed to create namespace: %v", err))
		} else {
			cli.Success("Namespace created")
		}
	} else {
		cli.Success("Namespace exists")
	}

	// 5. Verify chart exists
	chartPath := defaultHelmChart
	if !filepath.IsAbs(chartPath) {
		absChart, err := filepath.Abs(chartPath)
		if err == nil {
			chartPath = absChart
		}
	}

	if _, err := os.Stat(chartPath); os.IsNotExist(err) {
		return fmt.Errorf("helm chart not found: %s", chartPath)
	}

	// Set release name
	if releaseName == "" {
		releaseName = defaultHelmRelease
	}

	// Check if release already exists
	checkCmd := exec.Command("helm", "list", "-n", namespace, "-q")
	output, _ := checkCmd.Output()
	releases := strings.Split(strings.TrimSpace(string(output)), "\n")
	releaseExists := false
	for _, r := range releases {
		if r == releaseName {
			releaseExists = true
			break
		}
	}

	// Build helm command
	var cmd *exec.Cmd
	var action string

	if releaseExists {
		cli.Info(fmt.Sprintf("Release '%s' already exists, upgrading...", releaseName))
		action = "upgrade"
		cmd = exec.Command("helm", "upgrade", releaseName, chartPath, "-n", namespace, "--wait")
	} else {
		cli.Info("Installing Helm release...")
		action = "install"
		cmd = exec.Command("helm", "install", releaseName, chartPath, "-n", namespace, "--wait", "--create-namespace")
	}

	// Add config file override if provided
	if configPath != "" {
		absConfigPath, err := filepath.Abs(configPath)
		if err == nil {
			// Check if config file exists
			if _, err := os.Stat(absConfigPath); err == nil {
				// Note: The Helm chart would need to support config file override
				// For now, we'll note that config should be embedded in values
				cli.Info(fmt.Sprintf("Note: Using chart default config (custom config at %s)", absConfigPath))
			}
		}
	}

	// Add custom --set values
	for _, setValue := range setValues {
		cmd.Args = append(cmd.Args, "--set", setValue)
	}

	// Set observability
	if !withObs {
		cmd.Args = append(cmd.Args, "--set", "config.observability.tracing.enabled=false")
	}

	// Set timeout
	cmd.Args = append(cmd.Args, "--timeout", "10m")

	cli.Info(fmt.Sprintf("Running: %s", strings.Join(cmd.Args, " ")))

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm %s failed: %w", action, err)
	}

	cli.Success(fmt.Sprintf("Helm release '%s' %sd successfully", releaseName, action))

	// Get service information
	cli.Info("Fetching service information...")
	svcCmd := exec.Command("kubectl", "get", "svc", "-n", namespace, "-l", "app.kubernetes.io/name=semantic-router")
	svcCmd.Stdout = os.Stdout
	svcCmd.Stderr = os.Stderr
	_ = svcCmd.Run()

	cli.Info("\nNext steps:")
	cli.Info(fmt.Sprintf("  Check status: helm status %s -n %s", releaseName, namespace))
	cli.Info(fmt.Sprintf("  Check pods: kubectl get pods -n %s -l app.kubernetes.io/name=semantic-router", namespace))
	cli.Info(fmt.Sprintf("  View logs: kubectl logs -n %s -l app.kubernetes.io/name=semantic-router", namespace))
	cli.Info(fmt.Sprintf("  Port forward: kubectl port-forward -n %s svc/%s 8080:8080", namespace, releaseName))

	return nil
}

// UndeployHelm removes Helm release
func UndeployHelm(namespace, releaseName string, wait bool) error {
	cli.Info("Removing Helm release...")

	// Check if helm exists
	if !commandExists("helm") {
		return fmt.Errorf("helm not found")
	}

	// Set release name
	if releaseName == "" {
		releaseName = defaultHelmRelease
	}

	// Check if release exists
	checkCmd := exec.Command("helm", "list", "-n", namespace, "-q")
	output, err := checkCmd.Output()
	if err != nil {
		return fmt.Errorf("failed to list releases: %w", err)
	}

	releases := strings.Split(strings.TrimSpace(string(output)), "\n")
	releaseExists := false
	for _, r := range releases {
		if r == releaseName {
			releaseExists = true
			break
		}
	}

	if !releaseExists {
		cli.Warning(fmt.Sprintf("Release '%s' not found in namespace '%s'", releaseName, namespace))
		return nil
	}

	// Uninstall release
	cli.Info(fmt.Sprintf("Uninstalling release '%s'...", releaseName))
	cmd := exec.Command("helm", "uninstall", releaseName, "-n", namespace)

	if wait {
		cmd.Args = append(cmd.Args, "--wait")
		cli.Info("Waiting for resources to be deleted...")
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm uninstall failed: %w", err)
	}

	// Wait for pods to terminate if requested
	if wait {
		cli.Info("Verifying cleanup...")
		timeout := 300 // 5 minutes
		cleaned := false

		for i := 0; i < timeout; i += 5 {
			time.Sleep(5 * time.Second)

			// Check for pods
			//nolint:gosec // G204: releaseName and namespace are from internal config
			checkCmd := exec.Command("kubectl", "get", "pods", "-n", namespace, "-l", "app.kubernetes.io/instance="+releaseName, "--no-headers")
			output, err := checkCmd.Output()

			if err != nil || len(output) == 0 {
				cleaned = true
				break
			}

			podCount := len(splitLines(string(output)))
			if podCount == 0 {
				cleaned = true
				break
			}

			if i%10 == 0 {
				cli.Info(fmt.Sprintf("Waiting for cleanup... (%ds/%ds, %d pods remaining)", i+5, timeout, podCount))
			}
		}

		if !cleaned {
			cli.Warning("Some resources may still be terminating")
		} else {
			cli.Success("All resources cleaned up")
		}
	}

	cli.Success(fmt.Sprintf("Helm release '%s' uninstalled", releaseName))
	return nil
}

// UpgradeHelmRelease upgrades an existing Helm release
func UpgradeHelmRelease(configPath, namespace, releaseName string, timeout int) error {
	cli.Info("Upgrading Helm release...")

	// Check if helm exists
	if !commandExists("helm") {
		return fmt.Errorf("helm not found. Please install Helm: https://helm.sh/docs/intro/install/")
	}

	// Set release name
	if releaseName == "" {
		releaseName = defaultHelmRelease
	}

	// Check if release exists
	checkCmd := exec.Command("helm", "list", "-n", namespace, "-q")
	output, err := checkCmd.Output()
	if err != nil {
		return fmt.Errorf("failed to list releases: %w", err)
	}

	releases := strings.Split(strings.TrimSpace(string(output)), "\n")
	releaseExists := false
	for _, r := range releases {
		if r == releaseName {
			releaseExists = true
			break
		}
	}

	if !releaseExists {
		cli.Warning(fmt.Sprintf("Release '%s' not found in namespace '%s'", releaseName, namespace))
		cli.Info("Use 'vsr deploy helm' to create a new deployment")
		return nil
	}

	// Verify chart exists
	chartPath := defaultHelmChart
	if !filepath.IsAbs(chartPath) {
		absChart, err := filepath.Abs(chartPath)
		if err == nil {
			chartPath = absChart
		}
	}

	if _, err := os.Stat(chartPath); os.IsNotExist(err) {
		return fmt.Errorf("helm chart not found: %s", chartPath)
	}

	// Build upgrade command
	cli.Info(fmt.Sprintf("Upgrading release '%s'...", releaseName))
	cmd := exec.Command("helm", "upgrade", releaseName, chartPath, "-n", namespace, "--wait")

	// Set timeout
	if timeout > 0 {
		cmd.Args = append(cmd.Args, "--timeout", fmt.Sprintf("%ds", timeout))
	} else {
		cmd.Args = append(cmd.Args, "--timeout", "5m")
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm upgrade failed: %w", err)
	}

	cli.Success(fmt.Sprintf("Helm release '%s' upgraded successfully", releaseName))

	// Check rollout status
	cli.Info("Checking deployment status...")
	//nolint:gosec // G204: releaseName and namespace are from internal config
	rolloutCmd := exec.Command("kubectl", "rollout", "status", "deployment/"+releaseName, "-n", namespace, "--timeout=60s")
	rolloutCmd.Stdout = os.Stdout
	rolloutCmd.Stderr = os.Stderr
	if err := rolloutCmd.Run(); err != nil {
		cli.Warning("Deployment rollout status check failed")
	}

	cli.Info(fmt.Sprintf("Check status: helm status %s -n %s", releaseName, namespace))
	return nil
}

// DetectHelmDeployment checks if a Helm deployment exists
func DetectHelmDeployment(namespace string) *DeploymentStatus {
	status := &DeploymentStatus{
		Type:      "helm",
		IsRunning: false,
	}

	if !commandExists("helm") {
		return status
	}

	// List releases in namespace
	cmd := exec.Command("helm", "list", "-n", namespace, "-q")
	output, err := cmd.Output()
	if err != nil || len(output) == 0 {
		return status
	}

	releases := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, release := range releases {
		if release == defaultHelmRelease || strings.Contains(release, "semantic-router") {
			status.IsRunning = true
			status.ReleaseName = release
			break
		}
	}

	return status
}
