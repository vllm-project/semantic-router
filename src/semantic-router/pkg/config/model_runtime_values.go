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

const (
	ModelRetryUnavailable = "unavailable"
	ModelRetryTimeout     = "timeout"
)

var modelRetryEvidenceOrder = []string{
	ModelRetryUnavailable,
	ModelRetryTimeout,
}

type ModelExecutionSettings struct {
	MaxRetries     int      `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`
	RetryOn        []string `yaml:"retry_on,omitempty" json:"retry_on,omitempty"`
	RequestTimeout string   `yaml:"request_timeout,omitempty" json:"request_timeout,omitempty"`
	StreamTimeout  string   `yaml:"stream_timeout,omitempty" json:"stream_timeout,omitempty"`
}

type ModelRuntimePricing struct {
	InputCostPerMillionTokens      *string `yaml:"input_cost_per_million_tokens,omitempty" json:"input_cost_per_million_tokens,omitempty"`
	OutputCostPerMillionTokens     *string `yaml:"output_cost_per_million_tokens,omitempty" json:"output_cost_per_million_tokens,omitempty"`
	CacheReadCostPerMillionTokens  *string `yaml:"cache_read_cost_per_million_tokens,omitempty" json:"cache_read_cost_per_million_tokens,omitempty"`
	CacheWriteCostPerMillionTokens *string `yaml:"cache_write_cost_per_million_tokens,omitempty" json:"cache_write_cost_per_million_tokens,omitempty"`
}

func validateModelExecutionAndPricing(
	path string,
	controlField string,
	retryField string,
	execution *ModelExecutionSettings,
	pricing *ModelRuntimePricing,
) error {
	if execution.MaxRetries < 0 || execution.MaxRetries > 5 {
		return fmt.Errorf("%s.%s.%s must be between 0 and 5", path, controlField, retryField)
	}
	if execution.RequestTimeout == "" {
		execution.RequestTimeout = defaultModelInvocationTimeout
	}
	if execution.StreamTimeout == "" {
		execution.StreamTimeout = defaultModelInvocationTimeout
	}
	for field, value := range map[string]string{
		"timeout.request": execution.RequestTimeout,
		"timeout.stream":  execution.StreamTimeout,
	} {
		duration, err := time.ParseDuration(value)
		if err != nil || duration < time.Second || duration > 24*time.Hour {
			return fmt.Errorf("%s.%s.%s must be a duration between 1s and 24h", path, controlField, field)
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
	// The caller validates an effective clone so inheritance never changes the
	// sparse public authoring document that is later exported.
	if pricing.CacheReadCostPerMillionTokens == nil {
		pricing.CacheReadCostPerMillionTokens = cloneStringPointer(pricing.InputCostPerMillionTokens)
	}
	if pricing.CacheWriteCostPerMillionTokens == nil {
		pricing.CacheWriteCostPerMillionTokens = cloneStringPointer(pricing.InputCostPerMillionTokens)
	}
	return nil
}

func compileModelControl(
	modelName string,
	control ModelControl,
	pricing ModelRuntimePricing,
) (ModelExecutionSettings, ModelRuntimePricing, error) {
	retry := ModelRetry{}
	if control.Retry != nil {
		retry.Count = control.Retry.Count
		retry.On = append([]string(nil), control.Retry.On...)
	}
	if err := normalizeModelRetry(modelName, &retry); err != nil {
		return ModelExecutionSettings{}, ModelRuntimePricing{}, err
	}
	timeout := ModelTimeout{}
	if control.Timeout != nil {
		timeout = *control.Timeout
	}
	execution := ModelExecutionSettings{
		MaxRetries: retry.Count, RetryOn: append([]string(nil), retry.On...),
		RequestTimeout: timeout.Request, StreamTimeout: timeout.Stream,
	}
	effectivePricing := cloneModelRuntimePricing(pricing)
	if err := validateModelExecutionAndPricing(
		"providers.models["+modelName+"]", "control", "retry.count", &execution, &effectivePricing,
	); err != nil {
		return ModelExecutionSettings{}, ModelRuntimePricing{}, err
	}
	return execution, effectivePricing, nil
}

func normalizeModelRetry(modelName string, retry *ModelRetry) error {
	path := fmt.Sprintf("providers.models[%s].control.retry", modelName)
	if retry.Count < 0 || retry.Count > 5 {
		return fmt.Errorf("%s.count must be between 0 and 5", path)
	}
	if retry.Count == 0 {
		if len(retry.On) != 0 {
			return fmt.Errorf("%s.on must be empty when count is 0", path)
		}
		return nil
	}
	if len(retry.On) == 0 {
		retry.On = []string{ModelRetryUnavailable}
		return nil
	}
	seen := make(map[string]struct{}, len(retry.On))
	for index, trigger := range retry.On {
		if strings.TrimSpace(trigger) != trigger || !isModelRetryEvidence(trigger) {
			return fmt.Errorf(
				"%s.on[%d] must be one of %q or %q",
				path, index, ModelRetryUnavailable, ModelRetryTimeout,
			)
		}
		if _, duplicate := seen[trigger]; duplicate {
			return fmt.Errorf("%s.on contains duplicate trigger %q", path, trigger)
		}
		seen[trigger] = struct{}{}
	}
	retry.On = retry.On[:0]
	for _, trigger := range modelRetryEvidenceOrder {
		if _, found := seen[trigger]; found {
			retry.On = append(retry.On, trigger)
		}
	}
	return nil
}

func isModelRetryEvidence(value string) bool {
	for _, candidate := range modelRetryEvidenceOrder {
		if value == candidate {
			return true
		}
	}
	return false
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

func cloneModelRuntimePricing(input ModelRuntimePricing) ModelRuntimePricing {
	return ModelRuntimePricing{
		InputCostPerMillionTokens:      cloneStringPointer(input.InputCostPerMillionTokens),
		OutputCostPerMillionTokens:     cloneStringPointer(input.OutputCostPerMillionTokens),
		CacheReadCostPerMillionTokens:  cloneStringPointer(input.CacheReadCostPerMillionTokens),
		CacheWriteCostPerMillionTokens: cloneStringPointer(input.CacheWriteCostPerMillionTokens),
	}
}

func modelRuntimeIsPriced(pricing ModelRuntimePricing) bool {
	return pricing.InputCostPerMillionTokens != nil ||
		pricing.OutputCostPerMillionTokens != nil ||
		pricing.CacheReadCostPerMillionTokens != nil ||
		pricing.CacheWriteCostPerMillionTokens != nil
}

func validateCanonicalBilling(canonical *CanonicalConfig) error {
	priced := false
	for _, model := range canonical.Providers.Models {
		priced = priced || modelRuntimeIsPriced(model.Pricing)
	}
	currency, configured := canonicalBillingCurrency(canonical)
	if configured && currency == "" {
		return fmt.Errorf("global.billing.currency is required when global.billing is configured")
	}
	if configured && !billingCurrencyPattern.MatchString(currency) {
		return fmt.Errorf("global.billing.currency must be an uppercase ISO-4217 code")
	}
	if priced && !configured {
		return fmt.Errorf("global.billing.currency is required when providers.models define pricing")
	}
	return nil
}

func canonicalBillingCurrency(canonical *CanonicalConfig) (string, bool) {
	if canonical == nil || canonical.Global == nil || canonical.Global.Billing == nil {
		return "", false
	}
	return canonical.Global.Billing.Currency, true
}
