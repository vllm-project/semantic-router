package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SupportedCanonicalVersion is the canonical config contract this build implements.
// Producers of canonical documents must stamp this value; consumers reject anything else.
const SupportedCanonicalVersion = "v0.3"

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

	if v != SupportedCanonicalVersion {
		return fmt.Errorf("version: unsupported %q, this build supports %q", version, SupportedCanonicalVersion)
	}

	return nil
}
