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

// validateCanonicalVersion gates canonical input before it is interpreted, so a
// document written for a different contract cannot be read as this one.
//
// An absent version is accepted and warned about: existing documents omit it, and
// requiring it needs a migration path (see #2326).
func validateCanonicalVersion(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("config cannot be nil")
	}

	v := strings.TrimSpace(canonical.Version)
	if v == "" {
		logging.Warnf("version: not set, interpreting as %q", CanonicalConfigVersion)
		return nil
	}

	if !slices.Contains(acceptedCanonicalVersions, v) {
		return fmt.Errorf("version: unsupported %q, this build reads %s",
			canonical.Version, strings.Join(acceptedCanonicalVersions, ", "))
	}

	return nil
}
