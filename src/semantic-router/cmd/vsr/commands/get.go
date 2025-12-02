package commands

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NewGetCmd creates the get command
func NewGetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get [models|categories|decisions|endpoints]",
		Short: "Get information about router resources",
		Long: `Retrieve and display information about configured resources.

Available resources:
  models      - List all configured models
  categories  - List all routing categories
  decisions   - List all routing decisions
  endpoints   - List all backend endpoints`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			resource := args[0]
			configPath := cmd.Parent().Flag("config").Value.String()

			cfg, err := config.Load(configPath)
			if err != nil {
				return fmt.Errorf("failed to load config: %w", err)
			}

			outputFormat := cmd.Parent().Flag("output").Value.String()

			switch resource {
			case "models":
				return displayModels(cfg, outputFormat)
			case "categories":
				return displayCategories(cfg, outputFormat)
			case "decisions":
				return displayDecisions(cfg, outputFormat)
			case "endpoints":
				return displayEndpoints(cfg, outputFormat)
			default:
				return fmt.Errorf("unknown resource: %s (valid options: models, categories, decisions, endpoints)", resource)
			}
		},
	}

	return cmd
}

func displayModels(cfg *config.RouterConfig, format string) error {
	switch format {
	case "json":
		return cli.PrintJSON(cfg.ModelConfig)
	case "yaml":
		return cli.PrintYAML(cfg.ModelConfig)
	}

	// Table format
	headers := []string{"Model Name", "Endpoints", "Pricing"}
	var rows [][]string

	for modelName, modelCfg := range cfg.ModelConfig {
		endpoints := "N/A"
		if len(modelCfg.PreferredEndpoints) > 0 {
			endpoints = fmt.Sprintf("%v", modelCfg.PreferredEndpoints)
		}

		pricing := "N/A"
		if modelCfg.Pricing.Currency != "" {
			pricing = fmt.Sprintf("%s %.2f/%.2f per 1M",
				modelCfg.Pricing.Currency,
				modelCfg.Pricing.PromptPer1M,
				modelCfg.Pricing.CompletionPer1M)
		}

		rows = append(rows, []string{modelName, endpoints, pricing})
	}

	cli.PrintTable(headers, rows)
	return nil
}

func displayCategories(cfg *config.RouterConfig, format string) error {
	switch format {
	case "json":
		return cli.PrintJSON(cfg.Categories)
	case "yaml":
		return cli.PrintYAML(cfg.Categories)
	}

	// Table format
	headers := []string{"Category", "Description", "MMLU Categories"}
	var rows [][]string

	for _, category := range cfg.Categories {
		rows = append(rows, []string{
			category.Name,
			category.Description,
			fmt.Sprintf("%v", category.MMLUCategories),
		})
	}

	cli.PrintTable(headers, rows)
	return nil
}

func displayDecisions(cfg *config.RouterConfig, format string) error {
	switch format {
	case "json":
		return cli.PrintJSON(cfg.Decisions)
	case "yaml":
		return cli.PrintYAML(cfg.Decisions)
	}

	// Table format
	headers := []string{"Decision", "Description", "Priority", "Models"}
	var rows [][]string

	for _, decision := range cfg.Decisions {
		var models []string
		for _, ref := range decision.ModelRefs {
			models = append(models, ref.Model)
		}

		rows = append(rows, []string{
			decision.Name,
			decision.Description,
			fmt.Sprintf("%d", decision.Priority),
			fmt.Sprintf("%v", models),
		})
	}

	cli.PrintTable(headers, rows)
	return nil
}

func displayEndpoints(cfg *config.RouterConfig, format string) error {
	switch format {
	case "json":
		return cli.PrintJSON(cfg.VLLMEndpoints)
	case "yaml":
		return cli.PrintYAML(cfg.VLLMEndpoints)
	}

	// Table format
	headers := []string{"Name", "Address", "Port", "Weight"}
	var rows [][]string

	for _, endpoint := range cfg.VLLMEndpoints {
		rows = append(rows, []string{
			endpoint.Name,
			endpoint.Address,
			fmt.Sprintf("%d", endpoint.Port),
			fmt.Sprintf("%d", endpoint.Weight),
		})
	}

	cli.PrintTable(headers, rows)
	return nil
}
