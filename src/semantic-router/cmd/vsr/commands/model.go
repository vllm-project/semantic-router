package commands

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/model"
)

// NewModelCmd creates the model command
func NewModelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Manage semantic router models",
		Long: `Manage models used by the vLLM Semantic Router.

This includes downloading, listing, validating, and removing models.

Examples:
  # List all models
  vsr model list

  # Show detailed info about a model
  vsr model info lora-intent-classifier

  # Validate a model
  vsr model validate lora-intent-classifier

  # Validate all models
  vsr model validate --all

  # Remove a model
  vsr model remove pii-classifier

  # Download models (currently uses Makefile)
  vsr model download`,
	}

	cmd.AddCommand(NewModelListCmd())
	cmd.AddCommand(NewModelInfoCmd())
	cmd.AddCommand(NewModelValidateCmd())
	cmd.AddCommand(NewModelRemoveCmd())
	cmd.AddCommand(NewModelDownloadCmd())

	return cmd
}

// NewModelListCmd creates the model list command
func NewModelListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all models",
		Long: `List all models (both downloaded and configured).

Shows model ID, name, type, size, and download status.

Examples:
  # List all models
  vsr model list

  # List only downloaded models
  vsr model list --downloaded

  # List with JSON output
  vsr model list --output json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			downloadedOnly, _ := cmd.Flags().GetBool("downloaded")
			outputFormat := cmd.Parent().Parent().Flag("output").Value.String()

			mgr := model.NewModelManager("./models")
			models, err := mgr.ListModels()
			if err != nil {
				return fmt.Errorf("failed to list models: %w", err)
			}

			// Filter by downloaded if flag set
			if downloadedOnly {
				filtered := []model.ModelInfo{}
				for _, m := range models {
					if m.Downloaded {
						filtered = append(filtered, m)
					}
				}
				models = filtered
			}

			if len(models) == 0 {
				cli.Warning("No models found")
				cli.Info("Download models with: make download-models")
				return nil
			}

			// Output based on format
			switch outputFormat {
			case "json":
				// JSON output
				fmt.Println("[")
				for i, m := range models {
					comma := ","
					if i == len(models)-1 {
						comma = ""
					}
					fmt.Printf("  {\"id\":\"%s\",\"name\":\"%s\",\"type\":\"%s\",\"purpose\":\"%s\",\"architecture\":\"%s\",\"size\":\"%s\",\"downloaded\":%t}%s\n",
						m.ID, m.Name, m.Type, m.Purpose, m.Architecture, model.FormatSize(m.Size), m.Downloaded, comma)
				}
				fmt.Println("]")
			case "yaml":
				// YAML output
				fmt.Println("models:")
				for _, m := range models {
					fmt.Printf("  - id: %s\n", m.ID)
					fmt.Printf("    name: %s\n", m.Name)
					fmt.Printf("    type: %s\n", m.Type)
					fmt.Printf("    purpose: %s\n", m.Purpose)
					fmt.Printf("    architecture: %s\n", m.Architecture)
					fmt.Printf("    size: %s\n", model.FormatSize(m.Size))
					fmt.Printf("    downloaded: %t\n", m.Downloaded)
				}
			default:
				// Table output (default)
				cli.Info("╔══════════════════════════════════════════════════════════════════════════╗")
				cli.Info("║                            Available Models                              ║")
				cli.Info("╠══════════════════════════════════════════════════════════════════════════╣")
				cli.Info(fmt.Sprintf("║ %-30s %-12s %-10s %-12s ║", "Model ID", "Type", "Purpose", "Size"))
				cli.Info("╠══════════════════════════════════════════════════════════════════════════╣")

				for _, m := range models {
					status := "✓"
					if !m.Downloaded {
						status = "✗"
					}
					cli.Info(fmt.Sprintf("║ %s %-28s %-12s %-10s %-12s ║",
						status, m.ID, m.Type, m.Purpose, model.FormatSize(m.Size)))
				}

				cli.Info("╚══════════════════════════════════════════════════════════════════════════╝")

				// Summary
				downloadedCount := 0
				for _, m := range models {
					if m.Downloaded {
						downloadedCount++
					}
				}
				cli.Info(fmt.Sprintf("\nTotal: %d models (%d downloaded)", len(models), downloadedCount))
			}

			return nil
		},
	}

	cmd.Flags().Bool("downloaded", false, "Show only downloaded models")

	return cmd
}

// NewModelInfoCmd creates the model info command
func NewModelInfoCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "info <model-id>",
		Short: "Show detailed model information",
		Long: `Show detailed information about a specific model.

Includes size, path, type, architecture, and purpose.

Examples:
  # Show info for a model
  vsr model info lora-intent-classifier

  # Show info with JSON output
  vsr model info pii-classifier --output json`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			modelID := args[0]
			outputFormat := cmd.Parent().Parent().Flag("output").Value.String()

			mgr := model.NewModelManager("./models")
			modelInfo, err := mgr.GetModelInfo(modelID)
			if err != nil {
				return fmt.Errorf("failed to get model info: %w", err)
			}

			// Output based on format
			switch outputFormat {
			case "json":
				fmt.Printf("{\"id\":\"%s\",\"name\":\"%s\",\"type\":\"%s\",\"purpose\":\"%s\",\"architecture\":\"%s\",\"path\":\"%s\",\"size\":\"%s\",\"size_bytes\":%d,\"downloaded\":%t}\n",
					modelInfo.ID, modelInfo.Name, modelInfo.Type, modelInfo.Purpose, modelInfo.Architecture, modelInfo.Path, model.FormatSize(modelInfo.Size), modelInfo.Size, modelInfo.Downloaded)
			case "yaml":
				fmt.Printf("id: %s\n", modelInfo.ID)
				fmt.Printf("name: %s\n", modelInfo.Name)
				fmt.Printf("type: %s\n", modelInfo.Type)
				fmt.Printf("purpose: %s\n", modelInfo.Purpose)
				fmt.Printf("architecture: %s\n", modelInfo.Architecture)
				fmt.Printf("path: %s\n", modelInfo.Path)
				fmt.Printf("size: %s\n", model.FormatSize(modelInfo.Size))
				fmt.Printf("size_bytes: %d\n", modelInfo.Size)
				fmt.Printf("downloaded: %t\n", modelInfo.Downloaded)
			default:
				// Table output
				cli.Info("╔══════════════════════════════════════════════════════════════════════════╗")
				cli.Info(fmt.Sprintf("║ Model: %-65s║", modelInfo.Name))
				cli.Info("╠══════════════════════════════════════════════════════════════════════════╣")
				cli.Info(fmt.Sprintf("║ %-20s %-52s║", "ID:", modelInfo.ID))
				cli.Info(fmt.Sprintf("║ %-20s %-52s║", "Type:", modelInfo.Type))
				cli.Info(fmt.Sprintf("║ %-20s %-52s║", "Purpose:", modelInfo.Purpose))
				cli.Info(fmt.Sprintf("║ %-20s %-52s║", "Architecture:", modelInfo.Architecture))
				cli.Info(fmt.Sprintf("║ %-20s %-52s║", "Size:", model.FormatSize(modelInfo.Size)))
				cli.Info(fmt.Sprintf("║ %-20s %-52t║", "Downloaded:", modelInfo.Downloaded))
				cli.Info("╠══════════════════════════════════════════════════════════════════════════╣")
				cli.Info(fmt.Sprintf("║ Path: %-66s║", truncateString(modelInfo.Path, 66)))
				cli.Info("╚══════════════════════════════════════════════════════════════════════════╝")
			}

			return nil
		},
	}
}

// NewModelValidateCmd creates the model validate command
func NewModelValidateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "validate [model-id]",
		Short: "Validate model integrity",
		Long: `Validate that a model is properly downloaded and contains all required files.

Checks for config.json and model weight files (pytorch_model.bin or model.safetensors).

Examples:
  # Validate a specific model
  vsr model validate lora-intent-classifier

  # Validate all models
  vsr model validate --all`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			validateAll, _ := cmd.Flags().GetBool("all")
			mgr := model.NewModelManager("./models")

			if validateAll {
				// Validate all models
				cli.Info("Validating all models...")
				results, err := mgr.ValidateAllModels()
				if err != nil {
					return fmt.Errorf("failed to validate models: %w", err)
				}

				hasErrors := false
				for modelID, validationErr := range results {
					if validationErr != nil {
						cli.Error(fmt.Sprintf("✗ %s: %v", modelID, validationErr))
						hasErrors = true
					} else {
						cli.Success(fmt.Sprintf("✓ %s: valid", modelID))
					}
				}

				if hasErrors {
					return fmt.Errorf("some models failed validation")
				}

				cli.Success(fmt.Sprintf("\nAll %d models are valid", len(results)))
				return nil
			}

			// Validate specific model
			if len(args) == 0 {
				return fmt.Errorf("model-id required (or use --all flag)")
			}

			modelID := args[0]
			cli.Info(fmt.Sprintf("Validating model: %s", modelID))

			if err := mgr.ValidateModel(modelID); err != nil {
				cli.Error(fmt.Sprintf("Validation failed: %v", err))
				return err
			}

			cli.Success(fmt.Sprintf("Model '%s' is valid", modelID))

			// Show what was checked
			cli.Info("\nChecked:")
			cli.Info("  ✓ Directory exists")
			cli.Info("  ✓ config.json present")
			cli.Info("  ✓ Model weights present")

			return nil
		},
	}

	cmd.Flags().Bool("all", false, "Validate all models")

	return cmd
}

// NewModelRemoveCmd creates the model remove command
func NewModelRemoveCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "remove <model-id>",
		Short: "Remove a downloaded model",
		Long: `Delete a model from disk to free up space.

Requires confirmation unless --force flag is used.

Examples:
  # Remove a model (with confirmation)
  vsr model remove pii-classifier

  # Remove without confirmation
  vsr model remove pii-classifier --force`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			modelID := args[0]
			force, _ := cmd.Flags().GetBool("force")

			mgr := model.NewModelManager("./models")

			// Get model info first
			modelInfo, err := mgr.GetModelInfo(modelID)
			if err != nil {
				return fmt.Errorf("failed to get model info: %w", err)
			}

			if !modelInfo.Downloaded {
				return fmt.Errorf("model is not downloaded: %s", modelID)
			}

			// Show what will be removed
			cli.Warning(fmt.Sprintf("This will remove model: %s", modelInfo.Name))
			cli.Info(fmt.Sprintf("Path: %s", modelInfo.Path))
			cli.Info(fmt.Sprintf("Size: %s", model.FormatSize(modelInfo.Size)))

			// Confirmation prompt unless force flag is set
			if !force {
				fmt.Print("\nAre you sure? (y/N): ")
				var response string
				_, _ = fmt.Scanln(&response)
				if response != "y" && response != "Y" {
					cli.Info("Removal cancelled")
					return nil
				}
			}

			// Remove the model
			cli.Info("Removing model...")
			if err := mgr.RemoveModel(modelID); err != nil {
				return fmt.Errorf("failed to remove model: %w", err)
			}

			cli.Success(fmt.Sprintf("Model '%s' removed successfully", modelID))
			cli.Info(fmt.Sprintf("Freed %s of disk space", model.FormatSize(modelInfo.Size)))

			return nil
		},
	}

	cmd.Flags().Bool("force", false, "Skip confirmation prompt")

	return cmd
}

// NewModelDownloadCmd creates the model download command
func NewModelDownloadCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download models",
		Long: `Download models for the semantic router.

Currently uses the Makefile 'download-models' command.
Future versions will support direct HuggingFace downloads.

Examples:
  # Download all configured models
  vsr model download

  # Download with verbose output
  vsr model download --verbose`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cli.Info("Downloading models...")
			cli.Warning("Model download currently uses 'make download-models'")
			cli.Info("Please run: make download-models")

			// In the future, this will implement direct downloads:
			// mgr := model.NewModelManager("./models")
			// progress := func(downloaded, total int64) {
			//     percentage := float64(downloaded) / float64(total) * 100
			//     cli.Info(fmt.Sprintf("Progress: %.1f%%", percentage))
			// }
			// return mgr.DownloadModel(modelID, progress)

			return fmt.Errorf("direct model download not yet implemented")
		},
	}

	return cmd
}

// truncateString truncates a string to maxLen characters
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		// Pad with spaces
		return s + strings.Repeat(" ", maxLen-len(s))
	}
	return s[:maxLen-3] + "..."
}
