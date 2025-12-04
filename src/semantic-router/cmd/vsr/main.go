package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/cmd/vsr/commands"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	// Version information (set by build flags)
	version   = "dev"
	gitCommit = "unknown"
	buildDate = "unknown"
)

func main() {
	// Initialize logging
	if _, err := logging.InitLoggerFromEnv(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
	}

	rootCmd := &cobra.Command{
		Use:   "vsr",
		Short: "vLLM Semantic Router Control CLI",
		Long: `vsr is a command-line tool for managing the vLLM Semantic Router.

It provides a unified interface for installing, configuring, deploying, and
managing the router across different environments (local, Docker, Kubernetes).

Common workflows:
  vsr init                    # Initialize a new configuration
  vsr config validate         # Validate your configuration
  vsr deploy docker           # Deploy using Docker Compose
  vsr status                  # Check router status
  vsr test-prompt "test"      # Send a test prompt

For detailed help on any command, use:
  vsr <command> --help`,
		Version: fmt.Sprintf("%s (commit: %s, built: %s)", version, gitCommit, buildDate),
	}

	// Global flags
	rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "Enable verbose output")
	rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format: table, json, yaml")

	// Add subcommands
	rootCmd.AddCommand(commands.NewConfigCmd())
	rootCmd.AddCommand(commands.NewGetCmd())
	rootCmd.AddCommand(commands.NewDeployCmd())
	rootCmd.AddCommand(commands.NewUndeployCmd())
	rootCmd.AddCommand(commands.NewUpgradeCmd())
	rootCmd.AddCommand(commands.NewStatusCmd())
	rootCmd.AddCommand(commands.NewLogsCmd())
	rootCmd.AddCommand(commands.NewModelCmd())
	rootCmd.AddCommand(commands.NewDebugCmd())
	rootCmd.AddCommand(commands.NewHealthCmd())
	rootCmd.AddCommand(commands.NewDiagnoseCmd())
	rootCmd.AddCommand(commands.NewDashboardCmd())
	rootCmd.AddCommand(commands.NewMetricsCmd())
	rootCmd.AddCommand(commands.NewTestCmd())
	rootCmd.AddCommand(commands.NewInstallCmd())
	rootCmd.AddCommand(commands.NewInitCmd())
	rootCmd.AddCommand(commands.NewCompletionCmd())

	// Execute
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
