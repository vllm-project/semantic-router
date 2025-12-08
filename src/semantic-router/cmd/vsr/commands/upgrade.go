package commands

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/deployment"
)

// NewUpgradeCmd creates the upgrade command
func NewUpgradeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "upgrade [local|docker|kubernetes|helm]",
		Short: "Upgrade router deployment to latest version",
		Long: `Upgrade the vLLM Semantic Router deployment to the latest version.

This command performs a rolling upgrade with minimal downtime:
  - local:      Rebuild binary and gracefully restart
  - docker:     Pull latest images and recreate containers
  - kubernetes: Apply updated manifests and rolling restart
  - helm:       Upgrade Helm release with latest chart

Examples:
  # Upgrade local deployment
  vsr upgrade local

  # Upgrade Docker deployment
  vsr upgrade docker

  # Upgrade Docker with observability
  vsr upgrade docker --with-observability

  # Upgrade Kubernetes deployment
  vsr upgrade kubernetes

  # Upgrade Kubernetes in specific namespace with wait
  vsr upgrade kubernetes --namespace production --wait

  # Force upgrade without confirmation
  vsr upgrade docker --force

  # Upgrade with custom timeout
  vsr upgrade kubernetes --timeout 10m`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			env := args[0]
			configPath := cmd.Parent().Flag("config").Value.String()
			withObs, _ := cmd.Flags().GetBool("with-observability")
			namespace, _ := cmd.Flags().GetString("namespace")
			releaseName, _ := cmd.Flags().GetString("release-name")
			force, _ := cmd.Flags().GetBool("force")
			wait, _ := cmd.Flags().GetBool("wait")
			timeoutStr, _ := cmd.Flags().GetString("timeout")

			// Parse timeout
			timeout, err := time.ParseDuration(timeoutStr)
			if err != nil {
				return fmt.Errorf("invalid timeout format: %s (use format like '5m', '300s')", timeoutStr)
			}

			// Confirmation prompt unless force flag is set
			if !force {
				cli.Warning(fmt.Sprintf("This will upgrade the %s deployment", env))
				cli.Info("The router will be temporarily unavailable during the upgrade")
				fmt.Print("Continue? (y/N): ")
				var response string
				_, _ = fmt.Scanln(&response)
				if response != "y" && response != "Y" {
					cli.Info("Upgrade cancelled")
					return nil
				}
			}

			switch env {
			case "local":
				return deployment.UpgradeLocal(configPath)
			case "docker":
				return deployment.UpgradeDocker(configPath, withObs)
			case "kubernetes":
				return deployment.UpgradeKubernetes(configPath, namespace, int(timeout.Seconds()), wait)
			case "helm":
				return deployment.UpgradeHelmRelease(configPath, namespace, releaseName, int(timeout.Seconds()))
			default:
				return fmt.Errorf("unknown environment: %s", env)
			}
		},
	}

	cmd.Flags().Bool("with-observability", true, "Include observability stack (Docker only)")
	cmd.Flags().String("namespace", "default", "Kubernetes namespace (Kubernetes/Helm only)")
	cmd.Flags().String("release-name", "", "Helm release name (default: semantic-router)")
	cmd.Flags().Bool("force", false, "Skip confirmation prompt")
	cmd.Flags().Bool("wait", false, "Wait for upgrade to complete (Kubernetes/Helm only)")
	cmd.Flags().String("timeout", "5m", "Timeout for upgrade operation (e.g., '5m', '300s')")

	return cmd
}
