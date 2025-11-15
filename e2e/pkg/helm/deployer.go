package helm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"
)

// Deployer handles Helm chart deployments
type Deployer struct {
	KubeConfig string
	Verbose    bool
}

// NewDeployer creates a new Helm deployer
func NewDeployer(kubeConfig string, verbose bool) *Deployer {
	return &Deployer{
		KubeConfig: kubeConfig,
		Verbose:    verbose,
	}
}

// Install installs a Helm chart
func (d *Deployer) Install(ctx context.Context, opts InstallOptions) error {
	d.log("Installing Helm chart: %s/%s", opts.Namespace, opts.ReleaseName)

	args := []string{
		"install", opts.ReleaseName, opts.Chart,
		"--namespace", opts.Namespace,
		"--create-namespace",
		"--kubeconfig", d.KubeConfig,
	}

	if opts.Version != "" {
		args = append(args, "--version", opts.Version)
	}

	for _, valuesFile := range opts.ValuesFiles {
		args = append(args, "-f", valuesFile)
	}

	for key, value := range opts.Set {
		args = append(args, "--set", fmt.Sprintf("%s=%s", key, value))
	}

	if opts.Wait {
		args = append(args, "--wait")
		if opts.Timeout != "" {
			args = append(args, "--timeout", opts.Timeout)
		}
	}

	cmd := exec.CommandContext(ctx, "helm", args...)
	if d.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install chart: %w", err)
	}

	d.log("Chart %s installed successfully", opts.ReleaseName)
	return nil
}

// Uninstall uninstalls a Helm release
func (d *Deployer) Uninstall(ctx context.Context, releaseName, namespace string) error {
	d.log("Uninstalling Helm release: %s/%s", namespace, releaseName)

	cmd := exec.CommandContext(ctx, "helm", "uninstall", releaseName,
		"--namespace", namespace,
		"--kubeconfig", d.KubeConfig)

	if d.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to uninstall release: %w", err)
	}

	d.log("Release %s uninstalled successfully", releaseName)
	return nil
}

// WaitForDeployment waits for a deployment to be ready
func (d *Deployer) WaitForDeployment(ctx context.Context, namespace, deploymentName string, timeout time.Duration) error {
	d.log("Waiting for deployment %s/%s to be ready", namespace, deploymentName)

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "kubectl", "wait",
		"--for=condition=Available",
		fmt.Sprintf("deployment/%s", deploymentName),
		"-n", namespace,
		"--timeout=600s",
		"--kubeconfig", d.KubeConfig)

	if d.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("deployment failed to become ready: %w", err)
	}

	d.log("Deployment %s is ready", deploymentName)
	return nil
}

func (d *Deployer) log(format string, args ...interface{}) {
	if d.Verbose {
		fmt.Printf("[Helm] "+format+"\n", args...)
	}
}

// InstallOptions contains options for installing Helm charts
type InstallOptions struct {
	// ReleaseName is the name of the Helm release
	ReleaseName string

	// Chart is the chart reference (can be a path or repo/chart)
	Chart string

	// Namespace is the Kubernetes namespace
	Namespace string

	// Version is the chart version
	Version string

	// ValuesFiles are paths to values files
	ValuesFiles []string

	// Set contains key-value pairs to set
	Set map[string]string

	// Wait waits for resources to be ready
	Wait bool

	// Timeout is the timeout for waiting
	Timeout string
}
