package commands

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/deployment"
)

// NewDeployCmd creates the deploy command
func NewDeployCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "deploy [local|docker|kubernetes|helm]",
		Short: "Deploy the router to specified environment",
		Long: `Deploy the vLLM Semantic Router to different environments.

Supported environments:
  local       - Run router as local process
  docker      - Deploy using Docker Compose
  kubernetes  - Deploy to Kubernetes cluster
  helm        - Deploy using Helm chart`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			env := args[0]
			configPath := cmd.Parent().Flag("config").Value.String()
			withObs, _ := cmd.Flags().GetBool("with-observability")
			namespace, _ := cmd.Flags().GetString("namespace")
			releaseName, _ := cmd.Flags().GetString("release-name")
			setValues, _ := cmd.Flags().GetStringArray("set")
			force, _ := cmd.Flags().GetBool("force")

			switch env {
			case "local":
				return deployment.DeployLocal(configPath, force)
			case "docker":
				return deployment.DeployDocker(configPath, withObs)
			case "kubernetes":
				return deployment.DeployKubernetes(configPath, namespace, withObs)
			case "helm":
				return deployment.DeployHelm(configPath, namespace, releaseName, withObs, setValues)
			default:
				return fmt.Errorf("unknown environment: %s", env)
			}
		},
	}

	cmd.Flags().Bool("with-observability", true, "Deploy with Grafana/Prometheus observability stack")
	cmd.Flags().String("namespace", "default", "Kubernetes namespace for deployment")
	cmd.Flags().String("release-name", "", "Helm release name (default: semantic-router)")
	cmd.Flags().StringArray("set", []string{}, "Set values for Helm chart (can be used multiple times)")
	cmd.Flags().Bool("dry-run", false, "Show commands without executing")
	cmd.Flags().Bool("force", false, "Force replacement of existing local deployment")

	return cmd
}

// NewUndeployCmd creates the undeploy command
func NewUndeployCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "undeploy [local|docker|kubernetes|helm]",
		Short: "Remove router deployment",
		Long: `Remove the vLLM Semantic Router deployment from the specified environment.

Examples:
  # Undeploy local router
  vsr undeploy local

  # Undeploy Docker deployment
  vsr undeploy docker

  # Undeploy Docker and remove volumes
  vsr undeploy docker --volumes

  # Undeploy Kubernetes and wait for cleanup
  vsr undeploy kubernetes --wait

  # Undeploy from specific namespace
  vsr undeploy kubernetes --namespace production --wait

  # Undeploy Helm release
  vsr undeploy helm --namespace production --wait`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			env := args[0]
			namespace, _ := cmd.Flags().GetString("namespace")
			removeVolumes, _ := cmd.Flags().GetBool("volumes")
			wait, _ := cmd.Flags().GetBool("wait")
			releaseName, _ := cmd.Flags().GetString("release-name")

			switch env {
			case "local":
				return deployment.UndeployLocal()
			case "docker":
				return deployment.UndeployDocker(removeVolumes)
			case "kubernetes":
				return deployment.UndeployKubernetes(namespace, wait)
			case "helm":
				return deployment.UndeployHelm(namespace, releaseName, wait)
			default:
				return fmt.Errorf("unknown environment: %s", env)
			}
		},
	}

	cmd.Flags().String("namespace", "default", "Kubernetes namespace")
	cmd.Flags().String("release-name", "", "Helm release name (default: semantic-router)")
	cmd.Flags().Bool("volumes", false, "Remove volumes (Docker only)")
	cmd.Flags().Bool("wait", false, "Wait for complete cleanup (Kubernetes/Helm only)")
	return cmd
}

// NewStartCmd creates the start command
func NewStartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "start",
		Short: "Start the router service",
		RunE: func(cmd *cobra.Command, args []string) error {
			cli.Warning("Not implemented: use 'vsr deploy' instead")
			return nil
		},
	}
}

// NewStopCmd creates the stop command
func NewStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop",
		Short: "Stop the router service",
		RunE: func(cmd *cobra.Command, args []string) error {
			cli.Warning("Not implemented: use 'vsr undeploy' instead")
			return nil
		},
	}
}

// NewRestartCmd creates the restart command
func NewRestartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "restart",
		Short: "Restart the router service",
		RunE: func(cmd *cobra.Command, args []string) error {
			cli.Warning("Not implemented: use 'vsr undeploy' then 'vsr deploy' instead")
			return nil
		},
	}
}
