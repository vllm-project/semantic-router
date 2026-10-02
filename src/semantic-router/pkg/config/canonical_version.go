package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// acceptedCanonicalVersions are the contracts this build reads, written one first.
// A bump keeps the outgoing contract here for one cycle; at most two entries, so a scan beats a map.
var acceptedCanonicalVersions = []string{CanonicalConfigVersion}

// AcceptedCanonicalVersions returns a copy of the readable contracts in preference order.
func AcceptedCanonicalVersions() []string {
	return slices.Clone(acceptedCanonicalVersions)
}

// ValidateRawCanonicalVersion gates version on the parsed YAML map, before any
// normalizer or known-field check can reject the document for another reason.
func ValidateRawCanonicalVersion(raw map[string]interface{}) error {
	value, present := raw["version"]
	if !present {
		warnAbsentCanonicalVersion()
		return nil
	}
	if value == nil {
		return fmt.Errorf("version: must be a string, got null")
	}
	version, ok := value.(string)
	if !ok {
		return fmt.Errorf("version: must be a string, got %T", value)
	}
	return checkCanonicalVersion(version)
}

// validateCanonicalVersion applies the same gate to a decoded struct built outside the YAML path.
func validateCanonicalVersion(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("config cannot be nil")
	}
	if canonical.Version == "" {
		return nil
	}
	return checkCanonicalVersion(canonical.Version)
}

// checkCanonicalVersion compares exactly: no trimming or case folding, only "" is the fallback.
func checkCanonicalVersion(version string) error {
	if version == "" {
		warnAbsentCanonicalVersion()
		return nil
	}
	if slices.Contains(acceptedCanonicalVersions, version) {
		return nil
	}
	return fmt.Errorf("version: unsupported %q, this build reads %s",
		version, strings.Join(acceptedCanonicalVersions, ", "))
}

// warnAbsentCanonicalVersion warns instead of failing until #2326 gives a migration path.
func warnAbsentCanonicalVersion() {
	logging.Warnf("version: not set, interpreting as %q", CanonicalConfigVersion)
}
