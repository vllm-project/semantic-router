package helm

import (
	"fmt"
	"os"
	"strings"
	"time"
)

const semanticRouterHelmTimeoutEnv = "E2E_SEMANTIC_ROUTER_HELM_TIMEOUT"

func installTimeoutForRelease(releaseName, configured string) (string, error) {
	if releaseName != "semantic-router" {
		return configured, nil
	}

	override := strings.TrimSpace(os.Getenv(semanticRouterHelmTimeoutEnv))
	if override == "" {
		return configured, nil
	}

	overrideDuration, err := time.ParseDuration(override)
	if err != nil {
		return "", fmt.Errorf("invalid %s value %q: %w", semanticRouterHelmTimeoutEnv, override, err)
	}

	if configured == "" {
		return override, nil
	}

	configuredDuration, err := time.ParseDuration(configured)
	if err != nil {
		return "", fmt.Errorf("invalid helm timeout %q for release %q: %w", configured, releaseName, err)
	}

	if overrideDuration > configuredDuration {
		return override, nil
	}
	return configured, nil
}
