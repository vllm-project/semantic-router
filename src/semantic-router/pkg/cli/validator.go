package cli

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ValidationError represents a configuration validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

// ValidateConfig performs semantic validation on the configuration
func ValidateConfig(cfg *config.RouterConfig) error {
	var validationErrors []ValidationError

	// Validate model consistency
	if err := validateModelConsistency(cfg); err != nil {
		var target ValidationError
		if errors.As(err, &target) {
			validationErrors = append(validationErrors, target)
		}
	}

	// Validate categories
	if err := validateCategories(cfg); err != nil {
		var target ValidationError
		if errors.As(err, &target) {
			validationErrors = append(validationErrors, target)
		}
	}

	// Validate category mapping path
	if err := validateCategoryMappingPath(cfg); err != nil {
		var target ValidationError
		if errors.As(err, &target) {
			validationErrors = append(validationErrors, target)
		}
	}

	// Validate jailbreak
	if err := validateJailbreak(cfg); err != nil {
		var target ValidationError
		if errors.As(err, &target) {
			validationErrors = append(validationErrors, target)
		}
	}

	// Validate PII
	if err := validatePII(cfg); err != nil {
		var target ValidationError
		if errors.As(err, &target) {
			validationErrors = append(validationErrors, target)
		}
	}

	if len(validationErrors) > 0 {
		return validationErrors[0] // Return first error
	}

	return nil
}

func validateModelConsistency(cfg *config.RouterConfig) error {
	// Check that all models referenced in decisions exist in model_config
	for _, decision := range cfg.Decisions {
		for _, modelRef := range decision.ModelRefs {
			if _, exists := cfg.ModelConfig[modelRef.Model]; !exists {
				return ValidationError{
					Field:   fmt.Sprintf("decisions.%s.modelRefs", decision.Name),
					Message: fmt.Sprintf("model '%s' not found in model_config", modelRef.Model),
				}
			}
		}
	}

	// Check that default_model exists
	if cfg.DefaultModel != "" {
		if _, exists := cfg.ModelConfig[cfg.DefaultModel]; !exists {
			return ValidationError{
				Field:   "default_model",
				Message: fmt.Sprintf("default model '%s' not found in model_config", cfg.DefaultModel),
			}
		}
	}

	return nil
}

func validateCategories(cfg *config.RouterConfig) error {
	if len(cfg.Categories) == 0 {
		return ValidationError{
			Field:   "categories",
			Message: "at least one category must be defined",
		}
	}

	for _, category := range cfg.Categories {
		if len(category.ModelScores) == 0 {
			return ValidationError{
				Field:   fmt.Sprintf("categories.%s", category.Name),
				Message: "model_scores must be defined for each category",
			}
		}
	}

	return nil
}

func validateCategoryMappingPath(cfg *config.RouterConfig) error {
	if cfg.CategoryMappingPath == "" {
		return ValidationError{
			Field:   "category_mapping_path",
			Message: "category_mapping_path must be defined",
		}
	}
	if _, err := os.Stat(cfg.CategoryMappingPath); os.IsNotExist(err) {
		return ValidationError{
			Field:   "category_mapping_path",
			Message: fmt.Sprintf("category_mapping.json file not found at %s", cfg.CategoryMappingPath),
		}
	}
	return nil
}

func validateJailbreak(cfg *config.RouterConfig) error {
	if cfg.PromptGuard.Enabled {
		if cfg.PromptGuard.JailbreakMappingPath == "" {
			return ValidationError{
				Field:   "prompt_guard.jailbreak_mapping_path",
				Message: "jailbreak_mapping_path must be defined when prompt_guard is enabled",
			}
		}
		if _, err := os.Stat(cfg.PromptGuard.JailbreakMappingPath); os.IsNotExist(err) {
			return ValidationError{
				Field:   "prompt_guard.jailbreak_mapping_path",
				Message: fmt.Sprintf("jailbreak_type_mapping.json file not found at %s", cfg.PromptGuard.JailbreakMappingPath),
			}
		}
	}

	return nil
}

func validatePII(cfg *config.RouterConfig) error {
	if cfg.PromptGuard.Enabled {
		if cfg.PIIMappingPath == "" {
			return ValidationError{
				Field:   "pii_mapping_path",
				Message: "pii_mapping_path must be defined when prompt_guard is enabled",
			}
		}
		if _, err := os.Stat(cfg.PIIMappingPath); os.IsNotExist(err) {
			return ValidationError{
				Field:   "pii_mapping_path",
				Message: fmt.Sprintf("pii_type_mapping.json file not found at %s", cfg.PIIMappingPath),
			}
		}
	}

	return nil
}

// ValidateEndpointReachability checks if endpoints are reachable
func ValidateEndpointReachability(endpoint string) error {
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	resp, err := client.Get(endpoint)
	if err != nil {
		return fmt.Errorf("endpoint not reachable: %w", err)
	}
	defer resp.Body.Close()

	return nil
}
