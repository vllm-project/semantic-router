package commands

import (
	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/deployment"
)

// NewStatusCmd creates the status command
func NewStatusCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Check router and components status",
		Long: `Display status information for the router and its components.

This command auto-detects all running deployments (local, docker, kubernetes, helm)
and displays their status, components, and endpoints.

Examples:
  # Check status in default namespace
  vsr status

  # Check status in specific namespace
  vsr status --namespace production`,
		RunE: func(cmd *cobra.Command, args []string) error {
			namespace, _ := cmd.Flags().GetString("namespace")
			return deployment.CheckStatus(namespace)
		},
	}

	cmd.Flags().String("namespace", "default", "Kubernetes namespace to check")

	return cmd
}

// NewLogsCmd creates the logs command
func NewLogsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Fetch router logs",
		Long: `Stream or fetch logs from the router service with auto-detection of deployment type.

Supports filtering by component, time range, and pattern matching.

Examples:
  # Fetch last 100 lines (auto-detect deployment)
  vsr logs

  # Follow logs in real-time
  vsr logs --follow

  # Show logs from specific deployment type
  vsr logs --env docker

  # Filter by component
  vsr logs --component router

  # Show logs from specific namespace
  vsr logs --namespace production

  # Show logs since a time
  vsr logs --since 10m

  # Filter logs by pattern
  vsr logs --grep error

  # Combine options
  vsr logs --follow --env kubernetes --namespace prod --component router --grep "ERROR"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			follow, _ := cmd.Flags().GetBool("follow")
			tail, _ := cmd.Flags().GetInt("tail")
			namespace, _ := cmd.Flags().GetString("namespace")
			deployType, _ := cmd.Flags().GetString("env")
			component, _ := cmd.Flags().GetString("component")
			since, _ := cmd.Flags().GetString("since")
			grep, _ := cmd.Flags().GetString("grep")

			return deployment.FetchLogs(follow, tail, namespace, deployType, component, since, grep)
		},
	}

	cmd.Flags().BoolP("follow", "f", false, "Follow log output")
	cmd.Flags().IntP("tail", "n", 100, "Number of lines to show from the end")
	cmd.Flags().String("namespace", "default", "Kubernetes namespace (for K8s/Helm deployments)")
	cmd.Flags().String("env", "", "Deployment type: local, docker, kubernetes, helm (auto-detect if empty)")
	cmd.Flags().String("component", "", "Filter by component name (e.g., router, envoy, grafana)")
	cmd.Flags().String("since", "", "Show logs since duration (e.g., 10m, 1h) or timestamp")
	cmd.Flags().String("grep", "", "Filter logs by pattern (uses grep)")

	return cmd
}
