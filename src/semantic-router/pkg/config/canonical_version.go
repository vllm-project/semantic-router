package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// acceptedCanonicalVersions are the contracts this build can read.
// CanonicalConfigVersion is the one it writes; a release that bumps it keeps the
// outgoing contract listed here for one cycle, so configs written for it keep
// loading instead of being rejected all at once.
var acceptedCanonicalVersions = []string{CanonicalConfigVersion}

// AcceptedCanonicalVersions returns the contracts this build reads, in
// preference order. Every reader that gates on the version derives its own rule
// from this list rather than restating CanonicalConfigVersion, so the generated
// JSON Schema and the Router cannot disagree about what loads.
func AcceptedCanonicalVersions() []string {
	return slices.Clone(acceptedCanonicalVersions)
}

// ValidateRawCanonicalVersion gates the top-level version straight off the parsed
// YAML map, before any normalizer, environment rewrite, or known-field check runs.
// A document written for a different contract has to fail on its version and not
// on a field that contract happened to spell differently.
func ValidateRawCanonicalVersion(raw map[string]interface{}) error {
	value, present := raw["version"]
	if !present {
		warnAbsentCanonicalVersion()
		return nil
	}
	version, ok := value.(string)
	if !ok {
		return fmt.Errorf("version: must be a string, got %T", value)
	}
	return checkCanonicalVersion(version)
}

// validateCanonicalVersion repeats the gate on the decoded struct. The loader has
// already run ValidateRawCanonicalVersion by this point; this keeps the contract
// attached to the type for any future caller that builds a CanonicalConfig
// without going through the YAML path.
func validateCanonicalVersion(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("config cannot be nil")
	}
	if canonical.Version == "" {
		return nil
	}
	return checkCanonicalVersion(canonical.Version)
}

// checkCanonicalVersion compares the version exactly. It is not trimmed or
// case-folded: " v0.3 " and "   " are not the supported contract, and treating
// them as one would accept documents the previous exact comparison rejected.
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

// warnAbsentCanonicalVersion reports the one input that is accepted without
// naming a contract. Existing documents omit the field, and requiring it needs a
// migration path (see #2326), so this warns rather than failing.
func warnAbsentCanonicalVersion() {
	logging.Warnf("version: not set, interpreting as %q", CanonicalConfigVersion)
}
