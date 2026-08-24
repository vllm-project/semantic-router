package config

import (
	"fmt"
	"math/big"
	"regexp"
	"strings"
	"time"
)

const defaultModelInvocationTimeout = "300s"

var (
	modelPricePattern      = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?$`)
	billingCurrencyPattern = regexp.MustCompile(`^[A-Z]{3}$`)
)

type ModelExecutionSettings struct {
	MaxRetries     int    `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`
	RequestTimeout string `yaml:"request_timeout,omitempty" json:"request_timeout,omitempty"`
	StreamTimeout  string `yaml:"stream_timeout,omitempty" json:"stream_timeout,omitempty"`
}

type ModelRuntimePricing struct {
	InputCostPerMillionTokens      *string `yaml:"input_cost_per_million_tokens,omitempty" json:"input_cost_per_million_tokens,omitempty"`
	OutputCostPerMillionTokens     *string `yaml:"output_cost_per_million_tokens,omitempty" json:"output_cost_per_million_tokens,omitempty"`
	CacheReadCostPerMillionTokens  *string `yaml:"cache_read_cost_per_million_tokens,omitempty" json:"cache_read_cost_per_million_tokens,omitempty"`
	CacheWriteCostPerMillionTokens *string `yaml:"cache_write_cost_per_million_tokens,omitempty" json:"cache_write_cost_per_million_tokens,omitempty"`
}

func validateModelRuntimeValues(path string, execution *ModelExecutionSettings, pricing *ModelRuntimePricing) error {
	if execution.MaxRetries < 0 || execution.MaxRetries > 5 {
		return fmt.Errorf("%s.execution.max_retries must be between 0 and 5", path)
	}
	if execution.RequestTimeout == "" {
		execution.RequestTimeout = defaultModelInvocationTimeout
	}
	if execution.StreamTimeout == "" {
		execution.StreamTimeout = defaultModelInvocationTimeout
	}
	for field, value := range map[string]string{
		"request_timeout": execution.RequestTimeout,
		"stream_timeout":  execution.StreamTimeout,
	} {
		duration, err := time.ParseDuration(value)
		if err != nil || duration < time.Second || duration > 24*time.Hour {
			return fmt.Errorf("%s.execution.%s must be a duration between 1s and 24h", path, field)
		}
	}
	for field, value := range map[string]**string{
		"input_cost_per_million_tokens":       &pricing.InputCostPerMillionTokens,
		"output_cost_per_million_tokens":      &pricing.OutputCostPerMillionTokens,
		"cache_read_cost_per_million_tokens":  &pricing.CacheReadCostPerMillionTokens,
		"cache_write_cost_per_million_tokens": &pricing.CacheWriteCostPerMillionTokens,
	} {
		if err := normalizeModelPrice(field, value); err != nil {
			return fmt.Errorf("%s.pricing.%w", path, err)
		}
	}
	// Cache pricing is effective Model-revision state, not a runtime fallback.
	// Persisting the inherited values makes standalone export and managed
	// snapshot compilation produce the same immutable price contract.
	if pricing.CacheReadCostPerMillionTokens == nil {
		pricing.CacheReadCostPerMillionTokens = cloneStringPointer(pricing.InputCostPerMillionTokens)
	}
	if pricing.CacheWriteCostPerMillionTokens == nil {
		pricing.CacheWriteCostPerMillionTokens = cloneStringPointer(pricing.InputCostPerMillionTokens)
	}
	return nil
}

func normalizeModelPrice(field string, target **string) error {
	if target == nil || *target == nil {
		return nil
	}
	value := **target
	if !modelPricePattern.MatchString(value) {
		return fmt.Errorf("%s must be a plain non-negative decimal", field)
	}
	parts := strings.SplitN(value, ".", 2)
	if len(parts) == 2 {
		if len(parts[1]) > 9 {
			return fmt.Errorf("%s supports at most 9 fractional digits", field)
		}
		parts[1] = strings.TrimRight(parts[1], "0")
		if parts[1] == "" {
			value = parts[0]
		} else {
			value = parts[0] + "." + parts[1]
		}
	}
	parsed, ok := new(big.Rat).SetString(value)
	if !ok || parsed.Cmp(big.NewRat(1_000_000, 1)) > 0 {
		return fmt.Errorf("%s must not exceed 1000000", field)
	}
	canonical := value
	*target = &canonical
	return nil
}

func cloneStringPointer(input *string) *string {
	if input == nil {
		return nil
	}
	value := *input
	return &value
}

func modelRuntimeIsPriced(pricing ModelRuntimePricing) bool {
	return pricing.InputCostPerMillionTokens != nil ||
		pricing.OutputCostPerMillionTokens != nil ||
		pricing.CacheReadCostPerMillionTokens != nil ||
		pricing.CacheWriteCostPerMillionTokens != nil
}

func validateCanonicalBilling(canonical *CanonicalConfig) error {
	priced := false
	for _, model := range canonical.Models {
		priced = priced || modelRuntimeIsPriced(model.RuntimePricing)
	}
	currency, configured := canonicalBillingCurrency(canonical)
	mode := ControlPlaneModeStandalone
	if canonical.Global != nil && strings.TrimSpace(canonical.Global.ControlPlane.Mode) != "" {
		mode = strings.TrimSpace(canonical.Global.ControlPlane.Mode)
	}
	if mode == ControlPlaneModeManaged && configured {
		return fmt.Errorf("global.billing.currency is standalone-only; managed mode takes currency from Namespace")
	}
	if configured && currency == "" {
		return fmt.Errorf("global.billing.currency is required when global.billing is configured")
	}
	if configured && !billingCurrencyPattern.MatchString(currency) {
		return fmt.Errorf("global.billing.currency must be an uppercase ISO-4217 code")
	}
	if mode == ControlPlaneModeStandalone && priced && !configured {
		return fmt.Errorf("global.billing.currency is required when standalone Models define pricing")
	}
	return nil
}

func canonicalBillingCurrency(canonical *CanonicalConfig) (string, bool) {
	if canonical == nil || canonical.Global == nil || canonical.Global.Billing == nil {
		return "", false
	}
	return canonical.Global.Billing.Currency, true
}
