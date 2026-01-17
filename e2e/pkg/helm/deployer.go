package helm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

	chart, cleanup, err := d.prepareLocalChartWithDeps(ctx, opts.Chart)
	if err != nil {
		return err
	}
	if cleanup != nil {
		defer cleanup()
	}

	args := []string{
		"install", opts.ReleaseName, chart,
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

func (d *Deployer) prepareLocalChartWithDeps(ctx context.Context, chartRef string) (string, func(), error) {
	// Remote charts (oci://, http(s)://, repo/chart) are handled by Helm directly.
	if strings.Contains(chartRef, "://") {
		return chartRef, nil, nil
	}

	stat, err := os.Stat(chartRef)
	if err != nil || !stat.IsDir() {
		// Not a local directory; let Helm handle it.
		return chartRef, nil, nil
	}

	// Copy the chart to a temp dir so dependency build doesn't dirty the working tree.
	tmpDir, err := os.MkdirTemp("", "semantic-router-helm-chart-*")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create temp dir for chart copy: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	tmpChart := filepath.Join(tmpDir, "chart")
	if err := copyDir(chartRef, tmpChart); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("failed to copy chart to temp dir: %w", err)
	}

	// Build dependencies for local chart (creates charts/ and Chart.lock in temp copy).
	cmd := exec.CommandContext(ctx, "helm", "dependency", "build", tmpChart)
	if d.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("failed to build chart dependencies: %w", err)
	}

	return tmpChart, cleanup, nil
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)

		info, err := d.Info()
		if err != nil {
			return err
		}

		if d.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}

		// Copy file contents
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}

		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()

		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
		if err != nil {
			return err
		}
		defer out.Close()

		if _, err := out.ReadFrom(in); err != nil {
			return err
		}
		return nil
	})
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

	// Convert timeout to seconds for kubectl
	timeoutSeconds := int(timeout.Seconds())
	cmd := exec.CommandContext(ctx, "kubectl", "wait",
		"--for=condition=Available",
		fmt.Sprintf("deployment/%s", deploymentName),
		"-n", namespace,
		fmt.Sprintf("--timeout=%ds", timeoutSeconds),
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
