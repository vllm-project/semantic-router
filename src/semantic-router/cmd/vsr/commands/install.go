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
  model_scores:
  - model: your-model
    score: 0.9
    use_reasoning: true
    reasoning_description: "Mathematical problems benefit from step-by-step reasoning"
    reasoning_effort: high
- name: coding
  description: "Programming and code generation"
  model_scores:
  - model: your-model
    score: 0.8
    use_reasoning: false

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
    model_id: "models/mom-domain-classifier"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
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
  model_scores:
  - model: your-model
    score: 0.7
    use_reasoning: false

default_model: your-model

# Classification models
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
`

const fullTemplate = `# vLLM Semantic Router - Comprehensive Configuration
# This template includes all available options for advanced configuration.
# Uncomment and modify options as needed for your use case.

# ==============================================================================
# BERT Model Configuration
# ==============================================================================
# The BERT model is used for semantic similarity matching.
# Different models have different accuracy vs. speed tradeoffs.
bert_model:
  # HuggingFace model ID for semantic embedding
  model_id: sentence-transformers/all-MiniLM-L12-v2

  # Similarity threshold (0.0 to 1.0)
  # Higher values = stricter matching (fewer routes)
  # Lower values = more permissive (more potential routes)
  threshold: 0.6

  # Use CPU instead of GPU (recommended for smaller models)
  use_cpu: true

# ==============================================================================
# vLLM Endpoints - Backend Model Servers
# ==============================================================================
# Define multiple vLLM endpoints for load balancing and failover
vllm_endpoints:
  # Primary endpoint
  - name: "primary-llm"
    address: "127.0.0.1"
    port: 11434
    weight: 2  # Higher weight = more requests routed here

  # Secondary endpoint for failover
  - name: "secondary-llm"
    address: "127.0.0.1"
    port: 11435
    weight: 1

# ==============================================================================
# Model Configuration
# ==============================================================================
# Configure each model with preferred endpoints and pricing
model_config:
  # Your model name
  "your-model":
    # Preferred endpoints for this model (in order of preference)
    preferred_endpoints: ["primary-llm", "secondary-llm"]

    # Pricing information for cost tracking
    pricing:
      currency: "USD"
      # Cost per 1 million input tokens
      prompt_per_1m: 0.50
      # Cost per 1 million output tokens
      completion_per_1m: 1.50

  # Example: Second model with different endpoints
  # "another-model":
  #   preferred_endpoints: ["secondary-llm"]
  #   pricing:
  #     currency: "USD"
  #     prompt_per_1m: 0.75
  #     completion_per_1m: 2.00

# ==============================================================================
# Semantic Cache Configuration (Optional)
# ==============================================================================
# Cache similar queries to reduce API calls and latency
# semantic_cache:
#   # Enable caching
#   enabled: true
#
#   # Backend type: "memory", "redis", "milvus"
#   backend_type: "memory"
#
#   # Similarity threshold for cache hits (0.0 to 1.0)
#   similarity_threshold: 0.8
#
#   # Maximum number of cached entries
#   max_entries: 10000
#
#   # Time to live in seconds (0 = no expiration)
#   ttl_seconds: 3600
#
#   # Eviction policy: "fifo", "lru"
#   eviction_policy: "lru"
#
#   # Redis configuration (if backend_type: "redis")
#   # redis:
#   #   host: "localhost"
#   #   port: 6379
#   #   database: 0
#
#   # Milvus configuration (if backend_type: "milvus")
#   # milvus:
#   #   host: "localhost"
#   #   port: 19530
#   #   database: "semantic_cache"

# ==============================================================================
# Categories and Classification
# ==============================================================================
# Define categories for intent classification and routing
categories:
  - name: math
    description: "Mathematics related queries"
    model_scores:
      - model: your-model
        score: 0.9
        # Enable extended reasoning for complex problems
        use_reasoning: true
        reasoning_description: "Mathematical problems benefit from step-by-step reasoning"
        # reasoning_effort options: "low", "medium", "high"
        reasoning_effort: high

  - name: coding
    description: "Programming and code generation"
    model_scores:
      - model: your-model
        score: 0.8
        use_reasoning: false

  - name: writing
    description: "Writing and content generation"
    model_scores:
      - model: your-model
        score: 0.85
        use_reasoning: false

  # Add more categories as needed
  # - name: general
  #   description: "General knowledge questions"
  #   model_scores:
  #     - model: your-model
  #       score: 0.7
  #       use_reasoning: false

# ==============================================================================
# Keyword Rules - Pattern Matching for Routing
# ==============================================================================
# Define keyword-based rules to complement semantic routing
keyword_rules:
  - name: math_keywords
    operator: "OR"
    keywords: ["math", "calculus", "algebra", "geometry", "equation"]

  - name: code_keywords
    operator: "OR"
    keywords: ["code", "python", "javascript", "function", "error"]

  - name: finance_keywords
    operator: "AND"
    keywords: ["investment", "stock", "portfolio"]

# ==============================================================================
# Routing Decisions - How to Route Queries
# ==============================================================================
# Combine rules and semantic matching to make routing decisions
decisions:
  - name: math_decision
    description: "Route mathematical queries"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "keyword"
          name: "math_keywords"
    modelRefs:
      - model: your-model
        use_reasoning: true

  - name: code_decision
    description: "Route coding queries"
    priority: 9
    rules:
      operator: "OR"
      conditions:
        - type: "keyword"
          name: "code_keywords"
    modelRefs:
      - model: your-model
        use_reasoning: false

# ==============================================================================
# Default Model
# ==============================================================================
# Fallback model for queries that don't match any specific category
default_model: your-model

# ==============================================================================
# Classification Models (Optional)
# ==============================================================================
# Use specialized models for specific classification tasks
classifier:
  # Category classification model
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"

  # PII (Personally Identifiable Information) detection model
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true

  # Jailbreak/prompt injection detection model (if available)
  # jailbreak_model:
  #   model_id: "models/jailbreak_classifier"
  #   use_modernbert: true
  #   threshold: 0.8
  #   use_cpu: true

# ==============================================================================
# Security and Safety Features
# ==============================================================================

# Prompt Guard - Detect and block jailbreak attempts
# prompt_guard:
#   enabled: true
#   use_modernbert: true
#   threshold: 0.7
#   use_cpu: true

# PII Protection - Detect and redact sensitive information
# pii_protection:
#   enabled: true
#   redact_pii: true
#   block_on_pii: false  # Set to true to block requests with PII

# ==============================================================================
# Logging and Observability
# ==============================================================================
# logging:
#   level: "info"  # debug, info, warning, error
#   format: "json"  # json, text
#   output: "stdout"  # stdout, file, both

# ==============================================================================
# Advanced Options (Expert Users Only)
# ==============================================================================

# Request timeout in seconds
# request_timeout: 60

# Maximum batch size for concurrent requests
# max_batch_size: 32

# Enable detailed logging of all routing decisions
# debug_routing: false

# Enable metrics collection
# metrics:
#   enabled: true
#   port: 9090
#   path: "/metrics"
`
