package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/cmd/vsr/commands"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func main() {
	// Initialize logging
	if _, err := logging.InitLoggerFromEnv(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
	}

	rootCmd := &cobra.Command{
		Use:   "vsr",
		Short: "vLLM Semantic Router Training CLI",
		Long: `vsr train is a command-line tool for training model selection algorithms.

Commands:
  vsr train              # Train model selection algorithms
  vsr test-selector      # Test model selector with a query`,
	}

	// Add training commands
	rootCmd.AddCommand(commands.GetTrainCmd())
	rootCmd.AddCommand(commands.NewTestSelectorCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
