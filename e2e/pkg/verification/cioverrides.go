// e2e/pkg/verification/cioverrides.go

package verification

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// ciOverridePattern matches explicit per-profile CI selection overrides in
// the integration workflow: shell assignments of the form
//
//	<PROFILE>_CI_TESTS="case-a,case-b,..."
//
// The owning profile is derived from the variable prefix (upper snake to
// lower kebab), e.g. ENVOY_AI_GATEWAY_CI_TESTS -> envoy-ai-gateway. Deriving
// the binding from the variable name keeps this loader convention-based
// rather than hand-maintained: a future override that follows the same
// convention is picked up (and gated) automatically, and a prefix that
// matches no registered profile fails Gate D loudly.
var ciOverridePattern = regexp.MustCompile(`(?m)^\s*([A-Z0-9_]+)_CI_TESTS="([^"]*)"`)

// CIOverride records one explicit CI selection override: the exact testcase
// list the workflow runs for a profile instead of the profile's full
// runtime GetTestCases selection.
type CIOverride struct {
	Profile   string   `json:"profile"`
	TestCases []string `json:"test_cases"`
}

// LoadCIOverrides parses the CI workflow at path and returns the explicit
// per-profile selection overrides, sorted by profile. An empty result is a
// legitimate state (no override in force: CI runs each profile's complete
// selection); a missing or unreadable workflow file is a hard error.
func LoadCIOverrides(path string) ([]CIOverride, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading CI workflow %s: %w", path, err)
	}

	matches := ciOverridePattern.FindAllStringSubmatch(string(raw), -1)
	overrides := make([]CIOverride, 0, len(matches))
	seen := make(map[string]bool, len(matches))
	for _, match := range matches {
		profile := strings.ReplaceAll(strings.ToLower(match[1]), "_", "-")
		if seen[profile] {
			return nil, fmt.Errorf("CI workflow %s declares multiple %s_CI_TESTS overrides for profile %q", path, match[1], profile)
		}
		seen[profile] = true
		overrides = append(overrides, CIOverride{
			Profile:   profile,
			TestCases: dedupSorted(splitCommaList(match[2])),
		})
	}
	sort.Slice(overrides, func(i, j int) bool { return overrides[i].Profile < overrides[j].Profile })
	return overrides, nil
}

// CIOverrideDrift is the per-profile diagnosis of an explicit CI override
// against the profile's runtime selection and its bounded exclusion table.
type CIOverrideDrift struct {
	// OmittedUnrecorded lists testcases selected by the profile, absent
	// from the explicit CI list, and not recorded in the exclusion table:
	// the silent-omission scenario (a testcase added to the profile but
	// never executed in CI, with no gate noticing).
	OmittedUnrecorded []string

	// StaleExclusions lists exclusion-table entries that no longer describe
	// an exclusion: the testcase is now on the CI list, or the profile no
	// longer selects it.
	StaleExclusions []string

	// Phantom lists CI-list entries the profile does not select at
	// runtime: names that CI asks for but the execution graph cannot
	// resolve to a selected testcase.
	Phantom []string
}

// DiagnoseCIOverrideDrift computes the drift between one profile's runtime
// selection, its explicit CI list, and its bounded exclusion table. It is a
// pure function over its inputs so the omission, stale, and phantom
// scenarios can be pinned by synthetic tests independently of the live
// registries.
func DiagnoseCIOverrideDrift(selected, ciList []string, exclusions map[string]CIExclusion) CIOverrideDrift {
	selectedSet := toSet(selected)
	ciSet := toSet(ciList)

	drift := CIOverrideDrift{
		OmittedUnrecorded: []string{},
		StaleExclusions:   []string{},
		Phantom:           []string{},
	}

	for name := range selectedSet {
		if ciSet[name] {
			continue
		}
		if _, recorded := exclusions[name]; !recorded {
			drift.OmittedUnrecorded = append(drift.OmittedUnrecorded, name)
		}
	}
	for name := range exclusions {
		if !selectedSet[name] || ciSet[name] {
			drift.StaleExclusions = append(drift.StaleExclusions, name)
		}
	}
	for name := range ciSet {
		if !selectedSet[name] {
			drift.Phantom = append(drift.Phantom, name)
		}
	}

	sort.Strings(drift.OmittedUnrecorded)
	sort.Strings(drift.StaleExclusions)
	sort.Strings(drift.Phantom)
	return drift
}

func splitCommaList(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func toSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
