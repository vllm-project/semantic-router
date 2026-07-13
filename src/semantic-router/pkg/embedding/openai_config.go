package embedding

import (
	"errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func validateOpenAICompatibleConfig(cfg OpenAICompatibleConfig) (string, string, error) {
	if err := validateOpenAICompatibleRequestPolicy(cfg); err != nil {
		return "", "", err
	}
	endpoint, err := embeddingsEndpoint(cfg.BaseURL, cfg.APIKeyEnv != "")
	if err != nil {
		return "", "", err
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		return "", "", fmt.Errorf("embedding endpoint model is required for backend %q", config.EmbeddingBackendOpenAICompatible)
	}
	if err := validateOpenAICompatibleDimensions(cfg); err != nil {
		return "", "", err
	}
	return endpoint, model, nil
}

func validateOpenAICompatibleDimensions(cfg OpenAICompatibleConfig) error {
	if cfg.ExpectedDimension > 0 && cfg.Dimensions > 0 && cfg.ExpectedDimension != cfg.Dimensions {
		return fmt.Errorf("embedding endpoint dimensions (%d) must match target_dimension (%d)", cfg.Dimensions, cfg.ExpectedDimension)
	}
	if cfg.Dimensions < 0 || cfg.ExpectedDimension < 0 {
		return fmt.Errorf("embedding endpoint dimensions must be non-negative")
	}
	return nil
}

func validateOpenAICompatibleRequestPolicy(cfg OpenAICompatibleConfig) error {
	if cfg.TimeoutSeconds < 0 {
		return fmt.Errorf("embedding endpoint timeout_seconds must be non-negative")
	}
	if cfg.TimeoutSeconds > config.EmbeddingEndpointMaxTimeoutSeconds {
		return fmt.Errorf("embedding endpoint timeout_seconds must not exceed %d", config.EmbeddingEndpointMaxTimeoutSeconds)
	}
	if cfg.MaxRetries < 0 {
		return fmt.Errorf("embedding endpoint max_retries must be non-negative")
	}
	if cfg.MaxRetries > config.EmbeddingEndpointMaxRetries {
		return fmt.Errorf("embedding endpoint max_retries must not exceed %d", config.EmbeddingEndpointMaxRetries)
	}
	if cfg.APIKeyEnv != "" && !config.IsValidEmbeddingAPIKeyEnv(cfg.APIKeyEnv) {
		return fmt.Errorf("embedding endpoint api_key_env must be %s when set", config.EmbeddingAPIKeyEnvName)
	}
	return nil
}

func embeddingsEndpoint(baseURL string, requireHTTPS bool) (string, error) {
	parsed, err := parseEmbeddingBaseURL(baseURL, requireHTTPS)
	if err != nil {
		return "", err
	}
	path := strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(path, "/embeddings") {
		path += "/embeddings"
	}
	parsed.Path = path
	parsed.RawPath = ""
	return parsed.String(), nil
}

func parseEmbeddingBaseURL(baseURL string, requireHTTPS bool) (*url.URL, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return nil, fmt.Errorf("embedding endpoint base_url is required for backend %q", config.EmbeddingBackendOpenAICompatible)
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("embedding endpoint base_url must be a valid URL")
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if problem := embeddingBaseURLProblem(parsed, baseURL, requireHTTPS); problem != "" {
		return nil, errors.New(problem)
	}
	return parsed, nil
}

func embeddingBaseURLProblem(parsed *url.URL, rawURL string, requireHTTPS bool) string {
	if parsed.Scheme == "" || parsed.Host == "" {
		return "embedding endpoint base_url must include scheme and host"
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "embedding endpoint base_url must use http or https"
	}
	if parsed.User != nil {
		return "embedding endpoint base_url must not include userinfo credentials"
	}
	if hasEmbeddingURLQueryOrFragment(parsed, rawURL) {
		return "embedding endpoint base_url must not include query or fragment components"
	}
	if requireHTTPS && parsed.Scheme != "https" {
		return "embedding endpoint base_url must use https when api_key_env is set"
	}
	return ""
}

func hasEmbeddingURLQueryOrFragment(parsed *url.URL, rawURL string) bool {
	return parsed.RawQuery != "" || parsed.ForceQuery || strings.Contains(rawURL, "#")
}
