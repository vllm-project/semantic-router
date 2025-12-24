package deployment

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/timeout"
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
	var clusterInfoErr error
	func() {
		// 10 second timeout for initial cluster connection check
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		clusterInfoCmd := exec.CommandContext(ctx, "kubectl", "cluster-info")
		clusterInfoErr = clusterInfoCmd.Run()
	}()
	if clusterInfoErr != nil {
		if errors.Is(clusterInfoErr, context.DeadlineExceeded) {
			cli.Error("kubectl cluster-info timed out after 10 seconds. Your Kubernetes cluster may be unreachable.")
			return fmt.Errorf("kubectl cluster-info timed out: %w", clusterInfoErr)
		}
		cli.Error("Unable to connect to Kubernetes cluster")
		cli.Info("Check your kubeconfig: kubectl config view")
		return fmt.Errorf("no connection to Kubernetes cluster: %w", clusterInfoErr)
	}
	cli.Success("Cluster connection verified")

	// 4. Check/create namespace
	cli.Info(fmt.Sprintf("Checking namespace '%s'...", namespace))
	var nsCheckErr error
	func() {
		// 10 second timeout for namespace check
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		nsCheckCmd := exec.CommandContext(ctx, "kubectl", "get", "namespace", namespace)
		nsCheckErr = nsCheckCmd.Run()
	}()
	if nsCheckErr != nil {
		if errors.Is(nsCheckErr, context.DeadlineExceeded) {
			return fmt.Errorf("kubectl get namespace timed out after 10 seconds: %w", nsCheckErr)
		}
		cli.Info(fmt.Sprintf("Creating namespace '%s'...", namespace))
		var nsCreateErr error
		func() {
			// 10 second timeout for namespace creation
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			nsCreateCmd := exec.CommandContext(ctx, "kubectl", "create", "namespace", namespace)
			nsCreateCmd.Stdout = os.Stdout
			nsCreateCmd.Stderr = os.Stderr
			nsCreateErr = nsCreateCmd.Run()
		}()
		if nsCreateErr != nil {
			if errors.Is(nsCreateErr, context.DeadlineExceeded) {
				return fmt.Errorf("kubectl create namespace timed out after 10 seconds: %w", nsCreateErr)
			}
			cli.Warning(fmt.Sprintf("Failed to create namespace: %v", nsCreateErr))
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
	var output []byte
	var helmListErr error
	func() {
		// 10 second timeout for listing helm releases
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		checkCmd := exec.CommandContext(ctx, "helm", "list", "-n", namespace, "-q")
		output, helmListErr = checkCmd.Output()
	}()
	if helmListErr != nil {
		if errors.Is(helmListErr, context.DeadlineExceeded) {
			cli.Warning("helm list timed out after 10 seconds, proceeding with install...")
		} else {
			cli.Warning("Could not check for existing helm releases, proceeding with install...")
		}
	}
	releases := strings.Split(strings.TrimSpace(string(output)), "\n")
	releaseExists := false
	for _, r := range releases {
		if r == releaseName {
			releaseExists = true
			break
		}
	}

	// Build helm command
	var cmdArgs []string
	var action string

	if releaseExists {
		cli.Info(fmt.Sprintf("Release '%s' already exists, upgrading...", releaseName))
		action = "upgrade"
		cmdArgs = []string{"helm", "upgrade", releaseName, chartPath, "-n", namespace, "--wait"}
	} else {
		cli.Info("Installing Helm release...")
		action = "install"
		cmdArgs = []string{"helm", "install", releaseName, chartPath, "-n", namespace, "--wait", "--create-namespace"}
	}

	// Add config file override if provided
	if configPath != "" {
		absConfigPath, err := filepath.Abs(configPath)
		if err == nil {
			if _, err := os.Stat(absConfigPath); err == nil {
				cli.Info(fmt.Sprintf("Note: Using chart default config (custom config at %s)", absConfigPath))
			}
		}
	}

	// Add custom --set values
	for _, setValue := range setValues {
		cmdArgs = append(cmdArgs, "--set", setValue)
	}

	// Set observability
	if !withObs {
		cmdArgs = append(cmdArgs, "--set", "config.observability.tracing.enabled=false")
	}

	// Set timeout
	cmdArgs = append(cmdArgs, "--timeout", "10m")

	cli.Info(fmt.Sprintf("Running: %s", strings.Join(cmdArgs, " ")))

	//nolint:gosec // G204: cmdArgs are constructed from validated inputs
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	helmErr := timeout.RunCommandWithIdleTimeoutContext(
		context.Background(),
		cmd,
		timeout.DefaultConfig,
		fmt.Sprintf("helm %s", action),
	)

	if helmErr != nil {
		if timeout.IsIdleTimeout(helmErr) {
			msg := fmt.Sprintf("helm operation failed. Troubleshooting:\n  - Check cluster: kubectl cluster-info\n  - Verify namespace: kubectl get namespaces\n  - Check Helm status: helm list -n %s", namespace)
			return fmt.Errorf("%s: %w", msg, helmErr)
		}
		return fmt.Errorf("helm %s failed: %w", action, helmErr)
	}

	cli.Success(fmt.Sprintf("Helm release '%s' %sd successfully", releaseName, action))

	// Get service information
	cli.Info("Fetching service information...")
	var svcErr error
	func() {
		// 10 second timeout for getting service info
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		svcCmd := exec.CommandContext(ctx, "kubectl", "get", "svc", "-n", namespace, "-l", "app.kubernetes.io/name=semantic-router")
		svcCmd.Stdout = os.Stdout
		svcCmd.Stderr = os.Stderr
		svcErr = svcCmd.Run()
	}()
	if svcErr != nil {
		if errors.Is(svcErr, context.DeadlineExceeded) {
			cli.Warning("Could not get service info: command timed out after 10 seconds")
		}
	}

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
			var checkOutput []byte
			var checkErr error
			func() {
				// 10 second timeout for getting pod status
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				//nolint:gosec // G204: releaseName and namespace are from internal config
				checkCmd := exec.CommandContext(ctx, "kubectl", "get", "pods", "-n", namespace, "-l", "app.kubernetes.io/instance="+releaseName, "--no-headers")
				checkOutput, checkErr = checkCmd.Output()
			}()

			if checkErr != nil {
				if errors.Is(checkErr, context.DeadlineExceeded) {
					cli.Info("kubectl get pods timed out...")
				}
				cleaned = true
				break
			}

			if len(checkOutput) == 0 {
				cleaned = true
				break
			}

			podCount := len(splitLines(string(checkOutput)))
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
	cmdArgs := []string{"helm", "upgrade", releaseName, chartPath, "-n", namespace, "--wait"}

	// Set timeout for the helm command itself
	if timeout > 0 {
		cmdArgs = append(cmdArgs, "--timeout", fmt.Sprintf("%ds", timeout))
	} else {
		cmdArgs = append(cmdArgs, "--timeout", "5m")
	}

	var upgradeErr error
	func() {
		// Use a slightly larger timeout for the context (+30s) to allow Helm to handle
		// its own timeout first. This ensures we get Helm's more descriptive error messages
		// rather than a generic context deadline exceeded error.
		ctxTimeout := time.Duration(timeout+30) * time.Second
		if timeout == 0 {
			ctxTimeout = 5*time.Minute + 30*time.Second
		}
		ctx, cancel := context.WithTimeout(context.Background(), ctxTimeout)
		defer cancel()
		//nolint:gosec // G204: cmdArgs are constructed from validated inputs
		cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		upgradeErr = cmd.Run()
	}()

	if upgradeErr != nil {
		if errors.Is(upgradeErr, context.DeadlineExceeded) {
			return fmt.Errorf("helm upgrade timed out: %w", upgradeErr)
		}
		return fmt.Errorf("helm upgrade failed: %w", upgradeErr)
	}

	cli.Success(fmt.Sprintf("Helm release '%s' upgraded successfully", releaseName))

	// Check rollout status
	cli.Info("Checking deployment status...")
	var rolloutErr error
	func() {
		// 60 second timeout for rollout status check
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		//nolint:gosec // G204: releaseName and namespace are from internal config
		rolloutCmd := exec.CommandContext(ctx, "kubectl", "rollout", "status", "deployment/"+releaseName, "-n", namespace, "--timeout=60s")
		rolloutCmd.Stdout = os.Stdout
		rolloutCmd.Stderr = os.Stderr
		rolloutErr = rolloutCmd.Run()
	}()
	if rolloutErr != nil {
		if errors.Is(rolloutErr, context.DeadlineExceeded) {
			cli.Warning("Rollout status check timed out after 60 seconds")
		} else {
			cli.Warning("Deployment rollout status check failed")
		}
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
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "helm", "list", "-n", namespace, "-q")
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
