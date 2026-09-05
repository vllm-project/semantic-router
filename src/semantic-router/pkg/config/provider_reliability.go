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
	if err := validateProviderOutlierDetection(modelName, reliability); err != nil {
		return err
	}
	if err := validateProviderHealthCheck(modelName, reliability); err != nil {
		return err
	}
	return validateProviderTimeouts(modelName, reliability)
}

func validateProviderRetry(modelName string, reliability ProviderReliability) error {
	if reliability.RetryCount < 0 || reliability.RetryCount > 5 {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_count must be between 0 and 5",
			modelName,
		)
	}
	if reliability.RetryCount > 0 && strings.TrimSpace(reliability.RetryOn) == "" {
		return fmt.Errorf(
			"providers.models[%s].reliability.retry_on is required when retries are enabled",
			modelName,
		)
	}
	return nil
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

func parseOptionalDuration(field, modelName, raw string) (time.Duration, bool, error) {
	if raw == "" {
		return 0, false, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, false, fmt.Errorf("providers.models[%s].reliability.%s is invalid: %w", modelName, field, err)
	}
	if d < 0 {
		return 0, false, fmt.Errorf("providers.models[%s].reliability.%s cannot be negative", modelName, field)
	}
	return d, true, nil
}

func validateProviderTimeouts(
	modelName string,
	reliability ProviderReliability,
) error {
	streamIdleDur, hasStreamIdle, err := parseOptionalDuration("stream_idle_timeout", modelName, reliability.StreamIdleTimeout)
	if err != nil {
		return err
	}

	reqDur, hasReq, err := parseOptionalDuration("request_timeout", modelName, reliability.RequestTimeout)
	if err != nil {
		return err
	}
	if hasReq && reqDur == 0 && (!hasStreamIdle || streamIdleDur <= 0) {
		return fmt.Errorf(
			"providers.models[%s].reliability.request_timeout cannot be 0s without a positive stream_idle_timeout",
			modelName,
		)
	}

	if reliability.ConnectTimeout != "" {
		connDur, connErr := time.ParseDuration(reliability.ConnectTimeout)
		if connErr != nil {
			return fmt.Errorf(
				"providers.models[%s].reliability.connect_timeout is invalid: %w",
				modelName,
				connErr,
			)
		}
		if connDur <= 0 {
			return fmt.Errorf(
				"providers.models[%s].reliability.connect_timeout must be greater than 0",
				modelName,
			)
		}
	}

	return nil
}
