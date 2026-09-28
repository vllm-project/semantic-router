package config

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

var currencyCodePattern = regexp.MustCompile(`^[A-Z]{3}$`)

// validateModelPricingContracts keeps operator-supplied cost metadata safe for
// accounting and cost-aware selection. Pricing remains deployment metadata on
// providers.models[]; routing model cards do not own provider rates.
func validateModelPricingContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for modelName, params := range cfg.ModelConfig {
		if err := validateModelPricing(modelName, params.Pricing); err != nil {
			return err
		}
	}
	return nil
}

func validateModelPricing(modelName string, pricing ModelPricing) error {
	currency := strings.TrimSpace(pricing.Currency)
	if currency != "" && !currencyCodePattern.MatchString(currency) {
		return fmt.Errorf(
			"providers.models[%s].pricing.currency must be a three-letter uppercase currency code, got %q",
			modelName,
			pricing.Currency,
		)
	}

	rates := []struct {
		name  string
		value float64
	}{
		{name: "prompt_per_1m", value: pricing.PromptPer1M},
		{name: "completion_per_1m", value: pricing.CompletionPer1M},
		{name: "cached_input_per_1m", value: pricing.CachedInputPer1M},
	}
	if pricing.CacheWritePer1M != nil {
		rates = append(rates, struct {
			name  string
			value float64
		}{name: "cache_write_per_1m", value: *pricing.CacheWritePer1M})
	}

	for _, rate := range rates {
		if math.IsNaN(rate.value) || math.IsInf(rate.value, 0) || rate.value < 0 {
			return fmt.Errorf(
				"providers.models[%s].pricing.%s must be a finite, non-negative per-million-token rate",
				modelName,
				rate.name,
			)
		}
	}
	return nil
}
