package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SupportedCanonicalVersion is the canonical config contract this build writes.
// Producers of canonical documents stamp this value.
const SupportedCanonicalVersion = "v0.3"

// acceptedCanonicalVersions are the contracts this build can read. A release that
// bumps SupportedCanonicalVersion keeps the outgoing contract listed here for one
// cycle, so configs written for it keep loading instead of being rejected at once.
var acceptedCanonicalVersions = []string{SupportedCanonicalVersion}

// validateCanonicalVersion gates canonical input before it is interpreted, so a
// document written for a different contract cannot be read as this one.
//
// An absent version is accepted and warned about: existing documents omit it, and
// requiring it needs a migration path (see #2326).
func validateCanonicalVersion(version string) error {
	v := strings.TrimSpace(version)
	if v == "" {
		logging.Warnf("version: not set, interpreting as %q", SupportedCanonicalVersion)
		return nil
	}

	if !slices.Contains(acceptedCanonicalVersions, v) {
		return fmt.Errorf("version: unsupported %q, this build reads %s",
			version, strings.Join(acceptedCanonicalVersions, ", "))
	}

	return nil
}
