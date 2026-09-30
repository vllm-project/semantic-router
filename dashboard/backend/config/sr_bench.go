package config

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

var srBenchTokenEnvPattern = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)

// ValidateSRBenchConfig accepts only a server-owned origin. Request URLs never
// select the upstream, and credentials belong in the process environment.
func ValidateSRBenchConfig(origin, tokenEnv string) error {
	parsed, err := url.Parse(origin)
	if err != nil || origin != strings.TrimSpace(origin) || parsed.Hostname() == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil ||
		parsed.Path != "" || parsed.RawPath != "" || parsed.RawQuery != "" ||
		parsed.ForceQuery || parsed.Fragment != "" || parsed.Opaque != "" {
		return fmt.Errorf("sr-bench URL must be an absolute http(s) origin without credentials, path, query, or fragment")
	}
	if !srBenchTokenEnvPattern.MatchString(tokenEnv) {
		return fmt.Errorf("sr-bench token reference must be an uppercase environment variable name")
	}
	return nil
}
