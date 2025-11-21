package cli

import (
	"fmt"
	"net/http"
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
	var errors []ValidationError

	// Validate model consistency
	if err := validateModelConsistency(cfg); err != nil {
		errors = append(errors, err.(ValidationError))
	}

	// Validate endpoint reachability (optional, can be slow)
	// Commented out for now as it makes validation slow
	// if err := validateEndpointReachability(cfg); err != nil {
	// 	errors = append(errors, err.(ValidationError))
	// }

	// Validate categories
	if err := validateCategories(cfg); err != nil {
		errors = append(errors, err.(ValidationError))
	}

	if len(errors) > 0 {
		return errors[0] // Return first error
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
