// e2e/pkg/verification/gates_ci_test.go

package verification

// Gate D below extends the derived execution contract with the fourth
// selection layer: explicit per-profile CI overrides declared in the
// integration workflow. Registration side effects for the live gate come
// from the blank imports in gates_test.go (both files link into the same
// test binary). The synthetic pin tests at the bottom exercise the pure
// drift diagnosis against fixed inputs, so the omission, stale, and phantom
// scenarios stay pinned regardless of live registry state.

import (
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// Gate D: explicit CI selection overrides are part of the derived contract.
//
// For every override the workflow declares, the set of testcases the profile
// selects at runtime but the override omits must equal the profile's
// KnownCIExclusions keys in both directions (the same ratchet discipline as
// Gate A), and the override must not name testcases the profile does not
// select (no allowlist, the same discipline as Gate B). An exclusion table
// for a profile with no override in force is stale as a whole.
func TestGateDCIOverridesMatchKnownExclusions(t *testing.T) {
	inventory := buildInventoryForTest(t)

	workflowPath := filepath.Join(repoRootForTest(t), ".github", "workflows", "integration-test-k8s.yml")
	overrides, err := LoadCIOverrides(workflowPath)
	if err != nil {
		t.Fatalf("loading explicit CI overrides: %v", err)
	}

	selections := make(map[string][]string, len(inventory.Profiles))
	for _, profile := range inventory.Profiles {
		selections[profile.Name] = profile.TestCases
	}

	overriddenProfiles := make(map[string]bool, len(overrides))
	for _, override := range overrides {
		overriddenProfiles[override.Profile] = true

		selected, registered := selections[override.Profile]
		if !registered {
			t.Errorf("CI workflow declares an override for profile %q which has no runtime registration; fix the variable prefix or register the profile", override.Profile)
			continue
		}

		drift := DiagnoseCIOverrideDrift(selected, override.TestCases, KnownCIExclusions[override.Profile])
		for _, name := range drift.OmittedUnrecorded {
			t.Errorf("profile %q selects testcase %q but the explicit CI list omits it and KnownCIExclusions records no bounded exclusion; add it to the CI list or record a disposition", override.Profile, name)
		}
		for _, name := range drift.StaleExclusions {
			t.Errorf("KnownCIExclusions entry %q/%q is stale: the testcase is now on the CI list or no longer selected by the profile; delete the resolved entry", override.Profile, name)
		}
		for _, name := range drift.Phantom {
			t.Errorf("explicit CI list for profile %q names %q which the profile does not select at runtime; fix the name or the profile selection", override.Profile, name)
		}
	}

	for profile := range KnownCIExclusions {
		if !overriddenProfiles[profile] {
			t.Errorf("KnownCIExclusions records exclusions for profile %q but the CI workflow declares no override for it; delete the stale table", profile)
		}
	}
}

// Every exclusion entry must carry exactly one bounded disposition: an
// owning issue XOR a documented rationale.
func TestGateDExclusionEntriesAreBounded(t *testing.T) {
	for profile, exclusions := range KnownCIExclusions {
		for name, exclusion := range exclusions {
			hasIssue := exclusion.Issue != 0
			hasRationale := exclusion.Rationale != ""
			if hasIssue == hasRationale {
				t.Errorf("KnownCIExclusions entry %q/%q must set exactly one of Issue or Rationale", profile, name)
			}
		}
	}
}

// Synthetic pin: the review's omission scenario. A testcase added to a
// profile but omitted from the explicit CI list, with no recorded
// exclusion, must surface as drift instead of passing silently.
func TestCIOverrideDriftPinsOmissionScenario(t *testing.T) {
	drift := DiagnoseCIOverrideDrift(
		[]string{"case-kept", "case-added-but-omitted"},
		[]string{"case-kept"},
		map[string]CIExclusion{},
	)
	assertDrift(t, drift, CIOverrideDrift{
		OmittedUnrecorded: []string{"case-added-but-omitted"},
		StaleExclusions:   []string{},
		Phantom:           []string{},
	})
}

// Synthetic pin: both stale-exclusion causes. An entry resolves when its
// testcase joins the CI list or when the profile stops selecting it; either
// way the stale entry must surface until deleted.
func TestCIOverrideDriftPinsStaleExclusions(t *testing.T) {
	drift := DiagnoseCIOverrideDrift(
		[]string{"case-kept", "case-promoted"},
		[]string{"case-kept", "case-promoted"},
		map[string]CIExclusion{
			"case-promoted":   {Issue: 1},
			"case-deselected": {Issue: 1},
		},
	)
	assertDrift(t, drift, CIOverrideDrift{
		OmittedUnrecorded: []string{},
		StaleExclusions:   []string{"case-deselected", "case-promoted"},
		Phantom:           []string{},
	})
}

// Synthetic pin: the phantom direction. A CI-list name the profile does not
// select must surface as drift with no allowlist.
func TestCIOverrideDriftPinsPhantom(t *testing.T) {
	drift := DiagnoseCIOverrideDrift(
		[]string{"case-kept"},
		[]string{"case-kept", "case-phantom"},
		map[string]CIExclusion{},
	)
	assertDrift(t, drift, CIOverrideDrift{
		OmittedUnrecorded: []string{},
		StaleExclusions:   []string{},
		Phantom:           []string{"case-phantom"},
	})
}

// Synthetic pin: a consistent contract diagnoses clean, including a
// recorded exclusion that is genuinely in force.
func TestCIOverrideDriftCleanContract(t *testing.T) {
	drift := DiagnoseCIOverrideDrift(
		[]string{"case-kept", "case-excluded"},
		[]string{"case-kept"},
		map[string]CIExclusion{
			"case-excluded": {Issue: 1},
		},
	)
	assertDrift(t, drift, CIOverrideDrift{
		OmittedUnrecorded: []string{},
		StaleExclusions:   []string{},
		Phantom:           []string{},
	})
}

func assertDrift(t *testing.T, got, want CIOverrideDrift) {
	t.Helper()
	sort.Strings(want.OmittedUnrecorded)
	sort.Strings(want.StaleExclusions)
	sort.Strings(want.Phantom)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("drift diagnosis mismatch:\n  got:  %+v\n  want: %+v", got, want)
	}
}
