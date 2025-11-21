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
		Use:   "deploy [local|docker|kubernetes]",
		Short: "Deploy the router to specified environment",
		Long: `Deploy the vLLM Semantic Router to different environments.

Supported environments:
  local       - Run router as local process
  docker      - Deploy using Docker Compose
  kubernetes  - Deploy to Kubernetes cluster`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			env := args[0]
			configPath := cmd.Parent().Flag("config").Value.String()
			withObs, _ := cmd.Flags().GetBool("with-observability")
			namespace, _ := cmd.Flags().GetString("namespace")

			switch env {
			case "local":
				return deployment.DeployLocal(configPath)
			case "docker":
				return deployment.DeployDocker(configPath, withObs)
			case "kubernetes":
				return deployment.DeployKubernetes(configPath, namespace, withObs)
			default:
				return fmt.Errorf("unknown environment: %s", env)
			}
		},
	}

	cmd.Flags().Bool("with-observability", true, "Deploy with Grafana/Prometheus observability stack")
	cmd.Flags().String("namespace", "default", "Kubernetes namespace for deployment")
	cmd.Flags().Bool("dry-run", false, "Show commands without executing")

	return cmd
}

// NewUndeployCmd creates the undeploy command
func NewUndeployCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "undeploy [local|docker|kubernetes]",
		Short: "Remove router deployment",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			env := args[0]
			namespace, _ := cmd.Flags().GetString("namespace")

			switch env {
			case "local":
				return deployment.UndeployLocal()
			case "docker":
				return deployment.UndeployDocker()
			case "kubernetes":
				return deployment.UndeployKubernetes(namespace)
			default:
				return fmt.Errorf("unknown environment: %s", env)
			}
		},
	}

	cmd.Flags().String("namespace", "default", "Kubernetes namespace")
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
