package commands

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/deployment"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/metrics"
)

// NewDashboardCmd creates the dashboard command
func NewDashboardCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "dashboard",
		Short: "Open router dashboard in browser",
		Long: `Open the router dashboard in your default web browser.

Auto-detects the dashboard URL based on your deployment type:
  - Docker: http://localhost:8700
  - Kubernetes: Port-forwards and opens dashboard
  - Helm: Port-forwards and opens dashboard
  - Local: http://localhost:8700 (if available)

Examples:
  # Open dashboard (auto-detect deployment)
  vsr dashboard

  # Open dashboard for specific namespace
  vsr dashboard --namespace production

  # Open without auto-launching browser
  vsr dashboard --no-open`,
		RunE: func(cmd *cobra.Command, args []string) error {
			namespace, _ := cmd.Flags().GetString("namespace")
			noOpen, _ := cmd.Flags().GetBool("no-open")

			cli.Info("Detecting dashboard deployment...")

			// Auto-detect deployment type
			deployType := detectActiveDeployment(namespace)

			if deployType == "" {
				cli.Warning("No active deployment detected")
				cli.Info("Deploy the router first with: vsr deploy [local|docker|kubernetes|helm]")
				return fmt.Errorf("no active deployment found")
			}

			cli.Info(fmt.Sprintf("Detected deployment type: %s", deployType))

			var dashboardURL string
			var portForwardCmd *exec.Cmd

			switch deployType {
			case "docker", "local":
				dashboardURL = "http://localhost:8700"
				cli.Info("Dashboard should be available at: " + dashboardURL)

			case "kubernetes", "helm":
				// Set up port forwarding
				cli.Info("Setting up port forwarding...")

				// Find dashboard pod
				dashboardURL = "http://localhost:8700"

				portForwardCmd = exec.Command("kubectl", "port-forward",
					"-n", namespace,
					"svc/semantic-router-dashboard",
					"8700:8700")

				// Start port-forward in background
				if err := portForwardCmd.Start(); err != nil {
					cli.Warning(fmt.Sprintf("Failed to start port-forward: %v", err))
					cli.Info("Try manually: kubectl port-forward -n " + namespace + " svc/semantic-router-dashboard 8700:8700")
					return err
				}

				// Give it a moment to establish
				time.Sleep(2 * time.Second)
				cli.Success("Port forwarding established")

				// Clean up on exit
				defer func() {
					if portForwardCmd != nil && portForwardCmd.Process != nil {
						_ = portForwardCmd.Process.Kill()
						cli.Info("Port forwarding stopped")
					}
				}()
			}

			// Open browser
			if !noOpen {
				cli.Info("Opening dashboard in browser...")
				if err := openBrowser(dashboardURL); err != nil {
					cli.Warning(fmt.Sprintf("Failed to open browser: %v", err))
					cli.Info("Please open manually: " + dashboardURL)
				} else {
					cli.Success("Dashboard opened!")
				}
			} else {
				cli.Info("Dashboard URL: " + dashboardURL)
			}

			// For K8s/Helm, keep port-forward alive (Issue #3: Add signal handling)
			if portForwardCmd != nil {
				cli.Info("\nPort forwarding active. Press Ctrl+C to stop.")

				// Handle interrupt signal for graceful shutdown
				sigChan := make(chan os.Signal, 1)
				signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

				// Wait for either process exit or interrupt signal
				done := make(chan error)
				go func() {
					done <- portForwardCmd.Wait()
				}()

				select {
				case <-sigChan:
					cli.Info("Stopping port forwarding...")
					if portForwardCmd.Process != nil {
						_ = portForwardCmd.Process.Kill()
					}
				case err := <-done:
					if err != nil {
						cli.Warning(fmt.Sprintf("Port forwarding exited with error: %v", err))
					}
				}
			}

			return nil
		},
	}

	// Get default from VSR_NAMESPACE env var if set, otherwise use "default"
	nsDefault := os.Getenv("_VSR_NAMESPACE_DEFAULT")
	if nsDefault == "" {
		nsDefault = "default"
	}
	cmd.Flags().String("namespace", nsDefault, "Kubernetes namespace (env: VSR_NAMESPACE)")
	cmd.Flags().Bool("no-open", false, "Don't open browser automatically")

	return cmd
}

// NewMetricsCmd creates the metrics command
func NewMetricsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "metrics",
		Short: "Display router metrics",
		Long: `Display key metrics for the router.

Shows:
  - Request counts
  - Latency statistics
  - Error rates
  - Model usage
  - Cost tracking (if configured)

Examples:
  # Show current metrics
  vsr metrics

  # Show metrics for specific time range
  vsr metrics --since 1h

  # Show metrics with auto-refresh
  vsr metrics --watch`,
		RunE: func(cmd *cobra.Command, args []string) error {
			since, _ := cmd.Flags().GetString("since")
			watch, _ := cmd.Flags().GetBool("watch")

			if watch {
				// Watch mode - refresh every 5 seconds
				cli.Info("Metrics (refreshing every 5s, Ctrl+C to stop)")
				cli.Info("")

				for {
					displayMetrics(since)
					time.Sleep(5 * time.Second)
					// Clear screen
					fmt.Print("\033[H\033[2J")
					cli.Info("Metrics (refreshing every 5s, Ctrl+C to stop)")
					cli.Info("")
				}
			} else {
				// One-time display
				displayMetrics(since)
			}

			return nil
		},
	}

	cmd.Flags().String("since", "5m", "Time range (e.g., 5m, 1h, 24h)")
	cmd.Flags().Bool("watch", false, "Auto-refresh metrics")

	return cmd
}

// detectActiveDeployment detects the active deployment type
func detectActiveDeployment(namespace string) string {
	// Check in order of specificity
	if status := deployment.DetectHelmDeployment(namespace); status != nil && status.IsRunning {
		return "helm"
	}
	if status := deployment.DetectKubernetesDeployment(namespace); status != nil && status.IsRunning {
		return "kubernetes"
	}
	if status := deployment.DetectDockerDeployment(); status != nil && status.IsRunning {
		return "docker"
	}
	if status := deployment.DetectLocalDeployment(); status != nil && status.IsRunning {
		return "local"
	}
	return ""
}

// openBrowser opens a URL in the default browser
func openBrowser(url string) error {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	return cmd.Start()
}

// displayMetrics displays metrics from Prometheus
func displayMetrics(since string) {
	// Try to detect metrics endpoint
	metricsURL := metrics.DetectMetricsEndpoint()

	if metricsURL == "" {
		// No metrics endpoint available
		data := &metrics.MetricsData{
			Available: false,
			Error:     "No metrics endpoint found. Is the router running?",
		}
		metrics.FormatMetricsTable(data, since)
		fmt.Println()
		cli.Info("Troubleshooting:")
		cli.Info("  1. Check if router is running: vsr status")
		cli.Info("  2. For local deployment: Metrics at http://localhost:9190/metrics")
		cli.Info("  3. For docker deployment: Metrics at http://localhost:9190/metrics")
		cli.Info("  4. Deploy with observability for enhanced metrics")
		return
	}

	// Collect metrics
	collector := metrics.NewCollector(metricsURL)
	data, err := collector.Collect()
	if err != nil {
		cli.Error(fmt.Sprintf("Failed to collect metrics: %v", err))
		return
	}

	// Display metrics
	metrics.FormatMetricsTable(data, since)
}
