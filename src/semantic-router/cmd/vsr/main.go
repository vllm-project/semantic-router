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
		Short: "vLLM Semantic Router CLI",
		Long: `vsr is a command-line tool for testing model selection algorithms.

Commands:
  vsr test-selector      # Test model selector with a query

For training and validation, use the scripts in src/training/ml_model_selection/:
  python train.py --data-file benchmark.jsonl --output-dir models/
  go run validate.go --models-dir .cache/ml-models`,
	}

	// Add commands
	rootCmd.AddCommand(commands.NewTestSelectorCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
