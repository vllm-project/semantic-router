package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// NewInstallCmd creates the install command
func NewInstallCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "install",
		Short: "Install vLLM Semantic Router",
		Long: `Guide for installing the router in your environment.

This command detects your environment and provides installation instructions.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cli.Warning("Installation Guide")
			fmt.Println("\nThe vsr CLI is already installed if you're running this command!")
			fmt.Println("\nTo install globally on Linux/macOS:")
			fmt.Println("  sudo cp bin/vsr /usr/local/bin/vsr")
			fmt.Println("  sudo chmod +x /usr/local/bin/vsr")
			fmt.Println("  # Or run: make install-cli")

			fmt.Println("\nTo deploy the router:")
			fmt.Println("  1. Initialize configuration: vsr init")
			fmt.Println("  2. Edit your config: vsr config edit")
			fmt.Println("  3. Deploy: vsr deploy [local|docker|kubernetes]")
			fmt.Println("\nFor detailed installation guides, see:")
			fmt.Println("  https://github.com/vllm-project/semantic-router/tree/main/website/docs/installation")
			return nil
		},
	}
}

// NewInitCmd creates the init command
func NewInitCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize a new configuration file",
		Long: `Create a new configuration file from a template.

Available templates:
  default  - Full-featured configuration with all options
  minimal  - Minimal configuration to get started
  full     - Comprehensive configuration with comments`,
		RunE: func(cmd *cobra.Command, args []string) error {
			output, _ := cmd.Flags().GetString("output")
			template, _ := cmd.Flags().GetString("template")

			return initializeConfig(output, template)
		},
	}

	cmd.Flags().String("output", "config/config.yaml", "Output path for the configuration file")
	cmd.Flags().String("template", "default", "Template to use: default, minimal, full")

	return cmd
}

func initializeConfig(outputPath, template string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Check if file exists
	if _, err := os.Stat(outputPath); err == nil {
		return fmt.Errorf("config file already exists at %s (use --output to specify different path)", outputPath)
	}

	// Get template content
	templateContent := getTemplate(template)

	// Write to file
	if err := os.WriteFile(outputPath, []byte(templateContent), 0o644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	cli.Success(fmt.Sprintf("Created configuration file: %s", outputPath))
	fmt.Println("\nNext steps:")
	fmt.Println("  1. Edit the configuration: vsr config edit")
	fmt.Println("  2. Validate your config: vsr config validate")
	fmt.Println("  3. Deploy the router: vsr deploy docker")

	return nil
}

func getTemplate(template string) string {
	switch template {
	case "minimal":
		return minimalTemplate
	case "full":
		return fullTemplate
	default:
		return defaultTemplate
	}
}

const defaultTemplate = `# vLLM Semantic Router Configuration

# BERT model for semantic similarity
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

# vLLM endpoints - your backend models
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 11434
    weight: 1

# Model configuration
model_config:
  "your-model":
    preferred_endpoints: ["endpoint1"]
    pricing:
      currency: "USD"
      prompt_per_1m: 0.50
      completion_per_1m: 1.50

# Categories (Metadata)
categories:
- name: math
  description: "Mathematics related queries"
- name: coding
  description: "Programming and code generation"

# Routing Rules
keyword_rules:
- name: math_keywords
  operator: "OR"
  keywords: ["math", "calculus", "algebra"]

# Routing Decisions
decisions:
- name: math_decision
  description: "Route math queries to model"
  priority: 10
  rules:
    operator: "AND"
    conditions:
    - type: "keyword"
      name: "math_keywords"
  modelRefs:
  - model: your-model
    use_reasoning: true

default_model: your-model

# Classification models
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true

# Security features (optional)
prompt_guard:
  enabled: false
  use_modernbert: true
  threshold: 0.7
  use_cpu: true

# Semantic caching (optional)
semantic_cache:
  enabled: false
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
  eviction_policy: "fifo"
`

const minimalTemplate = `# Minimal vLLM Semantic Router Configuration

bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 11434
    weight: 1

model_config:
  "your-model":
    preferred_endpoints: ["endpoint1"]

categories:
- name: general
  description: "General queries"

default_model: your-model
`

const fullTemplate = defaultTemplate // For now, full is same as default
