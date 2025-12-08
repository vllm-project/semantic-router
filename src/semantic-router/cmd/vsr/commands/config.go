package commands

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NewConfigCmd creates the config command
func NewConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage router configuration",
		Long: `View, edit, validate, and modify router configuration files.

The config command provides subcommands for managing your router's YAML configuration:
  view      - Display the current configuration
  edit      - Open configuration in your editor
  validate  - Validate configuration file syntax and semantics
  set       - Set specific configuration values
  get       - Retrieve specific configuration values`,
	}

	cmd.AddCommand(newConfigViewCmd())
	cmd.AddCommand(newConfigEditCmd())
	cmd.AddCommand(newConfigValidateCmd())
	cmd.AddCommand(newConfigSetCmd())
	cmd.AddCommand(newConfigGetCmd())

	return cmd
}

func newConfigViewCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "view",
		Short: "Display current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Parent().Flag("config").Value.String()

			// Read the config file
			data, err := os.ReadFile(configPath)
			if err != nil {
				return fmt.Errorf("failed to read config: %w", err)
			}

			outputFormat := cmd.Parent().Parent().Flag("output").Value.String()

			switch outputFormat {
			case "json":
				// Convert YAML to JSON for output
				var yamlData interface{}
				if err := yaml.Unmarshal(data, &yamlData); err != nil {
					return fmt.Errorf("failed to parse config: %w", err)
				}
				return cli.PrintJSON(yamlData)
			case "yaml", "table":
				// Just print the raw YAML
				fmt.Println(string(data))
				return nil
			default:
				return fmt.Errorf("unsupported output format: %s", outputFormat)
			}
		},
	}
}

func newConfigEditCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "edit",
		Short: "Edit configuration in your default editor",
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Parent().Flag("config").Value.String()

			editor := os.Getenv("EDITOR")
			if editor == "" {
				editor = "vi" // fallback to vi
			}

			editorCmd := exec.Command(editor, configPath)
			editorCmd.Stdin = os.Stdin
			editorCmd.Stdout = os.Stdout
			editorCmd.Stderr = os.Stderr

			if err := editorCmd.Run(); err != nil {
				return fmt.Errorf("failed to run editor: %w", err)
			}

			cli.Success(fmt.Sprintf("Configuration edited: %s", configPath))
			cli.Warning("Remember to validate your changes with: vsr config validate")
			return nil
		},
	}
}

func newConfigValidateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "validate",
		Short: "Validate configuration file",
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Parent().Flag("config").Value.String()

			// Parse the configuration
			cfg, err := config.Parse(configPath)
			if err != nil {
				cli.Error(fmt.Sprintf("Validation failed: %v", err))
				return err
			}

			// Perform additional semantic validation
			if err := cli.ValidateConfig(cfg); err != nil {
				cli.Error(fmt.Sprintf("Semantic validation failed: %v", err))
				return err
			}

			cli.Success(fmt.Sprintf("Configuration is valid: %s", configPath))
			return nil
		},
	}
}

func newConfigSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set <key> <value>",
		Short: "Set a configuration value",
		Args:  cobra.ExactArgs(2),
		Example: `  vsr config set bert_model.threshold 0.7
  vsr config set default_model my-model`,
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Parent().Flag("config").Value.String()
			key := args[0]
			value := args[1]

			// Read current config
			data, err := os.ReadFile(configPath)
			if err != nil {
				return fmt.Errorf("failed to read config: %w", err)
			}

			var configData map[string]interface{}
			if unmarshalErr := yaml.Unmarshal(data, &configData); unmarshalErr != nil {
				return fmt.Errorf("failed to parse config: %w", unmarshalErr)
			}

			// Set the value using dot notation
			if setErr := setNestedValue(configData, key, value); setErr != nil {
				return setErr
			}

			// Write back to file
			newData, err := yaml.Marshal(configData)
			if err != nil {
				return fmt.Errorf("failed to serialize config: %w", err)
			}

			if err := os.WriteFile(configPath, newData, 0o644); err != nil {
				return fmt.Errorf("failed to write config: %w", err)
			}

			cli.Success(fmt.Sprintf("Set %s = %s", key, value))
			cli.Warning("Validate changes with: vsr config validate")
			return nil
		},
	}
}

func newConfigGetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get <key>",
		Short: "Get a configuration value",
		Args:  cobra.ExactArgs(1),
		Example: `  vsr config get bert_model.threshold
  vsr config get default_model`,
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Parent().Flag("config").Value.String()
			key := args[0]

			// Read config
			data, err := os.ReadFile(configPath)
			if err != nil {
				return fmt.Errorf("failed to read config: %w", err)
			}

			var configData map[string]interface{}
			if unmarshalErr := yaml.Unmarshal(data, &configData); unmarshalErr != nil {
				return fmt.Errorf("failed to parse config: %w", unmarshalErr)
			}

			// Get the value
			value, err := getNestedValue(configData, key)
			if err != nil {
				return err
			}

			fmt.Printf("%s: %v\n", key, value)
			return nil
		},
	}
}

// Helper functions for nested key access
func setNestedValue(data map[string]interface{}, key string, value string) error {
	keys := strings.Split(key, ".")
	current := data

	for i := 0; i < len(keys)-1; i++ {
		if next, ok := current[keys[i]].(map[string]interface{}); ok {
			current = next
		} else {
			return fmt.Errorf("key not found: %s", strings.Join(keys[:i+1], "."))
		}
	}

	current[keys[len(keys)-1]] = value
	return nil
}

func getNestedValue(data map[string]interface{}, key string) (interface{}, error) {
	keys := strings.Split(key, ".")
	var current interface{} = data

	for _, k := range keys {
		if m, ok := current.(map[string]interface{}); ok {
			current = m[k]
		} else {
			return nil, fmt.Errorf("key not found: %s", key)
		}
	}

	return current, nil
}
