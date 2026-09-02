package config

import (
	"fmt"
	"net"
	"net/url"
	"regexp"
	"strings"
	"time"
)

const (
	defaultEvaluationLedgerTimeout = 30 * time.Second
	maxEvaluationLedgerTimeout     = 10 * time.Minute
	evaluationManagementKeyEnv     = "VLLM_SR_DASHBOARD_RECIPE_TOKEN"
)

var evaluationSecretEnvPattern = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)

// EvaluationServiceEndpointConfig is a server-owned, authenticated evidence
// source. The zero value means the source is intentionally unavailable.
// Secret values never belong in this type; APIKeyEnv is only a reference to a
// Dashboard process environment variable.
type EvaluationServiceEndpointConfig struct {
	URL       string
	APIKeyEnv string
	Timeout   time.Duration
}

func (endpoint EvaluationServiceEndpointConfig) Configured() bool {
	return endpoint != (EvaluationServiceEndpointConfig{})
}

type evaluationEndpointFlags struct {
	url        *string
	apiKeyEnv  *string
	timeoutRaw *string
}

func resolveEvaluationEndpoint(
	name, rawURL, apiKeyEnv, timeoutRaw string,
) (EvaluationServiceEndpointConfig, error) {
	if rawURL != strings.TrimSpace(rawURL) || apiKeyEnv != strings.TrimSpace(apiKeyEnv) ||
		timeoutRaw != strings.TrimSpace(timeoutRaw) {
		return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint values must not contain surrounding whitespace", name)
	}
	if rawURL == "" && apiKeyEnv == "" && timeoutRaw == "" {
		return EvaluationServiceEndpointConfig{}, nil
	}
	if rawURL == "" || apiKeyEnv == "" {
		return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint requires both URL and API key environment reference", name)
	}
	if err := validateEvaluationOrigin(rawURL); err != nil {
		return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint URL: %w", name, err)
	}
	if err := validateEvaluationSecretEnv(apiKeyEnv); err != nil {
		return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint API key: %w", name, err)
	}
	timeout := defaultEvaluationLedgerTimeout
	if timeoutRaw != "" {
		parsed, err := time.ParseDuration(timeoutRaw)
		if err != nil {
			return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint timeout: %w", name, err)
		}
		timeout = parsed
	}
	if timeout <= 0 || timeout > maxEvaluationLedgerTimeout {
		return EvaluationServiceEndpointConfig{}, fmt.Errorf("%s evaluation endpoint timeout must be greater than zero and at most %s", name, maxEvaluationLedgerTimeout)
	}
	return EvaluationServiceEndpointConfig{URL: rawURL, APIKeyEnv: apiKeyEnv, Timeout: timeout}, nil
}

func validateEvaluationRuntimeConfig(cfg *Config) error {
	if cfg.EvaluationDeploymentsDir != strings.TrimSpace(cfg.EvaluationDeploymentsDir) {
		return fmt.Errorf("evaluation deployments directory must not contain surrounding whitespace")
	}
	for name, ref := range map[string]string{
		"router": cfg.EvaluationRouterAPIKeyEnv,
		"envoy":  cfg.EvaluationEnvoyAPIKeyEnv,
	} {
		if ref == "" {
			continue
		}
		if ref != strings.TrimSpace(ref) {
			return fmt.Errorf("evaluation %s API key environment reference must not contain surrounding whitespace", name)
		}
		if err := validateEvaluationSecretEnv(ref); err != nil {
			return fmt.Errorf("evaluation %s API key: %w", name, err)
		}
	}
	if cfg.EvaluationRouterAPIKeyEnv == evaluationManagementKeyEnv {
		return fmt.Errorf("evaluation Router API key cannot reuse the Dashboard management credential")
	}
	if cfg.EvaluationRouterAPIKeyEnv != "" && cfg.EvaluationRouterAPIKeyEnv == cfg.EvaluationEnvoyAPIKeyEnv {
		return fmt.Errorf("router and Envoy evaluation credential references must be distinct")
	}

	endpoints := []struct {
		name   string
		config EvaluationServiceEndpointConfig
	}{
		{"agent task ledger", cfg.EvaluationAgentTaskLedger},
		{"fault recovery ledger", cfg.EvaluationFaultRecoveryLedger},
		{"hard policy ledger", cfg.EvaluationHardPolicyLedger},
		{"production experiment ledger", cfg.EvaluationProductionExperimentLedger},
	}
	originOwners := make(map[string]string, len(endpoints)+2)
	for name, origin := range map[string]string{"Router": cfg.RouterAPIURL, "Envoy": cfg.EnvoyURL} {
		if origin == "" {
			continue
		}
		key, err := evaluationOriginKey(origin)
		if err != nil {
			return fmt.Errorf("%s evaluation origin: %w", name, err)
		}
		originOwners[key] = name
	}
	credentialOwners := make(map[string]string, len(endpoints)+2)
	for name, ref := range map[string]string{
		"Router": cfg.EvaluationRouterAPIKeyEnv,
		"Envoy":  cfg.EvaluationEnvoyAPIKeyEnv,
	} {
		if ref != "" {
			credentialOwners[ref] = name
		}
	}
	for _, item := range endpoints {
		if err := validateEvaluationEndpointConfig(item.name, item.config); err != nil {
			return err
		}
		if !item.config.Configured() {
			continue
		}
		originKey, err := evaluationOriginKey(item.config.URL)
		if err != nil {
			return fmt.Errorf("%s evaluation endpoint URL: %w", item.name, err)
		}
		if owner, duplicate := originOwners[originKey]; duplicate {
			return fmt.Errorf("%s and %s evaluation origins must be distinct", owner, item.name)
		}
		originOwners[originKey] = item.name
		if owner, duplicate := credentialOwners[item.config.APIKeyEnv]; duplicate {
			return fmt.Errorf("%s and %s evaluation credential references must be distinct", owner, item.name)
		}
		credentialOwners[item.config.APIKeyEnv] = item.name
	}
	return nil
}

func validateEvaluationEndpointConfig(name string, endpoint EvaluationServiceEndpointConfig) error {
	if !endpoint.Configured() {
		return nil
	}
	if endpoint.URL == "" || endpoint.APIKeyEnv == "" {
		return fmt.Errorf("%s evaluation endpoint requires both URL and API key environment reference", name)
	}
	if err := validateEvaluationOrigin(endpoint.URL); err != nil {
		return fmt.Errorf("%s evaluation endpoint URL: %w", name, err)
	}
	if err := validateEvaluationSecretEnv(endpoint.APIKeyEnv); err != nil {
		return fmt.Errorf("%s evaluation endpoint API key: %w", name, err)
	}
	if endpoint.Timeout <= 0 || endpoint.Timeout > maxEvaluationLedgerTimeout {
		return fmt.Errorf(
			"%s evaluation endpoint timeout must be greater than zero and at most %s",
			name, maxEvaluationLedgerTimeout,
		)
	}
	return nil
}

func validateEvaluationOrigin(raw string) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("must be an absolute http(s) origin")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" ||
		parsed.Path != "" || parsed.RawPath != "" || parsed.String() != raw {
		return fmt.Errorf("must be an exact canonical origin without credentials, path, query, fragment, whitespace, or trailing slash")
	}
	return nil
}

func evaluationOriginKey(raw string) (string, error) {
	if err := validateEvaluationOrigin(raw); err != nil {
		return "", err
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse origin: %w", err)
	}
	scheme := strings.ToLower(parsed.Scheme)
	hostname := strings.ToLower(parsed.Hostname())
	port := parsed.Port()
	if port == "" {
		if scheme == "http" {
			port = "80"
		} else {
			port = "443"
		}
	}
	return scheme + "://" + net.JoinHostPort(hostname, port), nil
}

func validateEvaluationSecretEnv(value string) error {
	if !evaluationSecretEnvPattern.MatchString(value) {
		return fmt.Errorf("must be an uppercase environment variable name")
	}
	return nil
}
