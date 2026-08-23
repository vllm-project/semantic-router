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

// ProviderReliability describes endpoint-selection policy used by internal
// model-selection tools. Inference dispatch uses the compiled Model execution
// policy and does not read this type from human v0.4 YAML.
type ProviderReliability struct {
	LBPolicy            string `yaml:"lb_policy,omitempty"`
	RetryCount          int    `yaml:"retry_count,omitempty"`
	RetryOn             string `yaml:"retry_on,omitempty"`
	Consecutive5xx      int    `yaml:"consecutive_5xx,omitempty"`
	BaseEjectionTime    string `yaml:"base_ejection_time,omitempty"`
	MaxEjectionPercent  int    `yaml:"max_ejection_percent,omitempty"`
	HealthCheckPath     string `yaml:"health_check_path,omitempty"`
	HealthCheckInterval string `yaml:"health_check_interval,omitempty"`
	HealthCheckTimeout  string `yaml:"health_check_timeout,omitempty"`
}

func validateProviderReliability(modelName string, reliability ProviderReliability) error {
	switch strings.TrimSpace(reliability.LBPolicy) {
	case "", ProviderLBPolicyRoundRobin, ProviderLBPolicyLeastRequest:
	default:
		return fmt.Errorf(
			"model[%s].reliability.lb_policy must be %q or %q",
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
	return validateProviderHealthCheck(modelName, reliability)
}

func validateProviderRetry(modelName string, reliability ProviderReliability) error {
	if reliability.RetryCount < 0 || reliability.RetryCount > 5 {
		return fmt.Errorf(
			"model[%s].reliability.retry_count must be between 0 and 5",
			modelName,
		)
	}
	if reliability.RetryCount > 0 && strings.TrimSpace(reliability.RetryOn) == "" {
		return fmt.Errorf(
			"model[%s].reliability.retry_on is required when retries are enabled",
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
			"model[%s].reliability.consecutive_5xx cannot be negative",
			modelName,
		)
	}
	if reliability.MaxEjectionPercent < 0 || reliability.MaxEjectionPercent > 100 {
		return fmt.Errorf(
			"model[%s].reliability.max_ejection_percent must be between 0 and 100",
			modelName,
		)
	}
	if reliability.BaseEjectionTime != "" {
		if _, err := time.ParseDuration(reliability.BaseEjectionTime); err != nil {
			return fmt.Errorf(
				"model[%s].reliability.base_ejection_time is invalid: %w",
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
			"model[%s].reliability.health_check_path must start with /",
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
				"model[%s].reliability.%s is invalid: %w",
				modelName,
				field,
				err,
			)
		}
	}
	return nil
}
