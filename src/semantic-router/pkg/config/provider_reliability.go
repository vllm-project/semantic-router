package config

import (
	"fmt"
	"strings"
	"time"
)

const (
	ProviderLBPolicyRoundRobin   = "round_robin"
	ProviderLBPolicyLeastRequest = "least_request"
)

func validateProviderReliability(modelName string, reliability ProviderReliability) error {
	switch strings.TrimSpace(reliability.LBPolicy) {
	case "", ProviderLBPolicyRoundRobin, ProviderLBPolicyLeastRequest:
	default:
		return fmt.Errorf(
			"providers.models[%s].reliability.lb_policy must be %q or %q",
			modelName,
			ProviderLBPolicyRoundRobin,
			ProviderLBPolicyLeastRequest,
		)
	}
	if err := validateProviderRetry(modelName, reliability); err != nil {
		return err
	}
	if err := validateProviderTimeouts(modelName, reliability); err != nil {
		return err
	}
	if err := validateProviderOutlierDetection(modelName, reliability); err != nil {
		return err
	}
	return validateProviderHealthCheck(modelName, reliability)
}

// DefaultProviderRetryOn is the retry_on applied when retries are enabled
// without one, as the CLI renders it for Envoy.
const DefaultProviderRetryOn = "connect-failure,refused-stream"

func validateProviderRetry(modelName string, reliability ProviderReliability) error {
	if reliability.RetryCount < 0 || reliability.RetryCount > 5 {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_count must be between 0 and 5",
			modelName,
		)
	}
	for _, code := range reliability.RetriableStatusCodes {
		if code < 100 || code > 599 {
			return fmt.Errorf(
				"providers.models[%s].reliability.retriable_status_codes has %d; expected 100..599",
				modelName, code,
			)
		}
	}
	if reliability.RetryBudgetPercent < 0 || reliability.RetryBudgetPercent > 100 {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_budget_percent must be between 0 and 100",
			modelName,
		)
	}
	if reliability.RetryBudgetMinConcurrency < 0 {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_budget_min_concurrency cannot be negative",
			modelName,
		)
	}
	return validateProviderBackOff(modelName, reliability)
}

// DefaultProviderRetryBackOffBase is Envoy's default retry back-off base.
const DefaultProviderRetryBackOffBase = 25 * time.Millisecond

func validateProviderBackOff(modelName string, reliability ProviderReliability) error {
	base, err := providerDuration(modelName, "retry_back_off_base", reliability.RetryBackOffBase, false)
	if err != nil {
		return err
	}
	maximum, err := providerDuration(modelName, "retry_back_off_max", reliability.RetryBackOffMax, false)
	if err != nil {
		return err
	}
	if base == 0 {
		base = DefaultProviderRetryBackOffBase
	}
	if maximum != 0 && maximum < base {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_back_off_max must not be below retry_back_off_base",
			modelName,
		)
	}
	_, err = providerDuration(modelName, "retry_after_max", reliability.RetryAfterMax, false)
	return err
}

func validateProviderTimeouts(modelName string, reliability ProviderReliability) error {
	for _, field := range []struct {
		name, value string
		zeroOK      bool
	}{
		{"connect_timeout", reliability.ConnectTimeout, false},
		{"total_timeout", reliability.TotalTimeout, true},
		{"idle_timeout", reliability.IdleTimeout, true},
		{"per_try_timeout", reliability.PerTryTimeout, false},
		{"first_byte_timeout", reliability.FirstByteTimeout, false},
	} {
		if _, err := providerDuration(modelName, field.name, field.value, field.zeroOK); err != nil {
			return err
		}
	}
	return nil
}

func providerDuration(modelName, field, value string, zeroOK bool) (time.Duration, error) {
	return reliabilityDuration(fmt.Sprintf("providers.models[%s].reliability.%s", modelName, field), value, zeroOK)
}

// reliabilityDuration parses the optional reliability duration at path.
// Empty is zero; zeroOK admits an explicit 0 that disables the timeout.
func reliabilityDuration(path, value string, zeroOK bool) (time.Duration, error) {
	if strings.TrimSpace(value) == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(strings.TrimSpace(value))
	if err != nil || d < 0 || (d == 0 && !zeroOK) {
		return 0, fmt.Errorf("%s %q must be a positive duration such as 30s", path, value)
	}
	return d, nil
}

func validateProviderOutlierDetection(
	modelName string,
	reliability ProviderReliability,
) error {
	if reliability.Consecutive5xx < 0 {
		return fmt.Errorf(
			"providers.models[%s].reliability.consecutive_5xx cannot be negative",
			modelName,
		)
	}
	if reliability.MaxEjectionPercent < 0 || reliability.MaxEjectionPercent > 100 {
		return fmt.Errorf(
			"providers.models[%s].reliability.max_ejection_percent must be between 0 and 100",
			modelName,
		)
	}
	if reliability.BaseEjectionTime != "" {
		if _, err := time.ParseDuration(reliability.BaseEjectionTime); err != nil {
			return fmt.Errorf(
				"providers.models[%s].reliability.base_ejection_time is invalid: %w",
				modelName,
				err,
			)
		}
	}
	return nil
}

func validateProviderHealthCheck(
	modelName string,
	reliability ProviderReliability,
) error {
	if reliability.HealthCheckPath != "" &&
		!strings.HasPrefix(reliability.HealthCheckPath, "/") {
		return fmt.Errorf(
			"providers.models[%s].reliability.health_check_path must start with /",
			modelName,
		)
	}
	for field, value := range map[string]string{
		"health_check_interval": reliability.HealthCheckInterval,
		"health_check_timeout":  reliability.HealthCheckTimeout,
	} {
		if value == "" {
			continue
		}
		if _, err := time.ParseDuration(value); err != nil {
			return fmt.Errorf(
				"providers.models[%s].reliability.%s is invalid: %w",
				modelName,
				field,
				err,
			)
		}
	}
	return nil
}
