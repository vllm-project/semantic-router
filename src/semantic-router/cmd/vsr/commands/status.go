package commands

import (
	"github.com/spf13/cobra"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/deployment"
)

// NewStatusCmd creates the status command
func NewStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Check router and components status",
		Long:  `Display status information for the router and its components.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return deployment.CheckStatus()
		},
	}
}

// NewLogsCmd creates the logs command
func NewLogsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logs",
		Short: "Fetch router logs",
		Long:  `Stream or fetch logs from the router service.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			follow, _ := cmd.Flags().GetBool("follow")
			tail, _ := cmd.Flags().GetInt("tail")

			return deployment.FetchLogs(follow, tail)
		},
	}

	cmd.Flags().BoolP("follow", "f", false, "Follow log output")
	cmd.Flags().IntP("tail", "n", 100, "Number of lines to show from the end")

	return cmd
}
