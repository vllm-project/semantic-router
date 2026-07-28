package config

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v2"
)

// CanonicalVersion is the only steady-state router config version understood
// by this build. Older layouts must pass through explicit migration tooling.
const CanonicalVersion = "v0.3"

// SupportedCanonicalVersions returns the steady-state versions accepted by the
// router. Return a fresh slice so callers cannot mutate the package contract.
func SupportedCanonicalVersions() []string {
	return []string{CanonicalVersion}
}

// ValidateCanonicalConfig runs a typed producer's output through the same
// steady-state loader contract used for config files.
func ValidateCanonicalConfig(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("canonical config is nil")
	}
	data, err := yaml.Marshal(canonical)
	if err != nil {
		return fmt.Errorf("marshal canonical config: %w", err)
	}
	if _, err := ParseYAMLBytes(data); err != nil {
		return err
	}
	return nil
}

// ValidateCanonicalVersion rejects an absent, malformed, or unsupported
// version field before a boundary interprets the rest of a canonical document.
func ValidateCanonicalVersion(raw map[string]interface{}) error {
	value, ok := raw["version"]
	if !ok {
		return fmt.Errorf("version: required; supported versions: %s", CanonicalVersion)
	}

	version, ok := value.(string)
	if !ok {
		return fmt.Errorf("version: must be a string; supported versions: %s", CanonicalVersion)
	}
	version = strings.TrimSpace(version)
	if version == "" {
		return fmt.Errorf("version: must not be empty; supported versions: %s", CanonicalVersion)
	}
	if version != CanonicalVersion {
		return fmt.Errorf(
			"version: unsupported config version %q; supported versions: %s; older configs must be migrated explicitly with `vllm-sr config migrate`",
			version,
			CanonicalVersion,
		)
	}
	return nil
}
