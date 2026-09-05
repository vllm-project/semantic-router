// e2e/pkg/verification/inventory.go

// Package verification derives read-only views of the E2E execution graph
// from runtime sources of truth and gates their consistency.
//
// It OWNS: derived views of existing E2E registrations, reachability
// analysis, and consistency diagnostics.
//
// It DOES NOT OWN: testcase registration, profile registration, testcase
// selection, CI profile selection, or runtime surface catalogs. Arrows only
// point inward: this package reads the registries (testcases.List,
// framework.RegisteredProfileNames + Profile.GetTestCases) and never becomes
// a configuration owner. Callers that need registration side effects (the
// e2e-audit command and the gate tests) blank-import the registration
// packages themselves.
//
// The generated inventory is an output, never a checked-in source of truth.
// See issue #2379.
package verification

import (
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// ProfileCoverage records the testcases one canonical registered profile
// resolves to at runtime via GetTestCases (including shared testmatrix groups,
// which are expanded transitively by the profile implementations themselves).
type ProfileCoverage struct {
	Name      string   `json:"name"`
	TestCases []string `json:"test_cases"`
}

// TestcaseCoverage records which canonical profiles select one registered
// testcase. An empty Profiles list means the testcase is registered but
// unreachable from every registered profile.
type TestcaseCoverage struct {
	Name     string   `json:"name"`
	Profiles []string `json:"profiles"`
}

// Inventory is the runtime-derived execution graph.
type Inventory struct {
	Profiles  []ProfileCoverage  `json:"profiles"`
	Testcases []TestcaseCoverage `json:"testcases"`

	// SelectedButUnregistered lists names returned by some profile's
	// GetTestCases that no registered testcase carries (typos, stale
	// testmatrix entries, selections pointing at deleted registrations).
	SelectedButUnregistered []string `json:"selected_but_unregistered"`
}

// BuildInventory enumerates the registered testcases and canonical registered
// profiles and derives the execution graph. It requires the registration
// packages to have been imported by the caller; it fails loudly if either
// registry is empty rather than reporting a trivially empty graph.
func BuildInventory() (Inventory, error) {
	registered := registeredTestcaseNames()
	if len(registered) == 0 {
		return Inventory{}, fmt.Errorf("testcase registry is empty; the caller must blank-import the e2e/testcases registration package")
	}

	profileNames := framework.RegisteredProfileNames()
	if len(profileNames) == 0 {
		return Inventory{}, fmt.Errorf("profile registry is empty; the caller must blank-import the e2e/profiles/all registration package")
	}
	sort.Strings(profileNames)

	inventory := Inventory{
		Profiles:  make([]ProfileCoverage, 0, len(profileNames)),
		Testcases: make([]TestcaseCoverage, 0, len(registered)),
	}

	selectedBy := make(map[string][]string, len(registered))
	for _, profileName := range profileNames {
		profile, err := framework.NewProfileByName(profileName)
		if err != nil {
			return Inventory{}, fmt.Errorf("instantiating registered profile %q: %w", profileName, err)
		}
		selected := dedupSorted(profile.GetTestCases())
		inventory.Profiles = append(inventory.Profiles, ProfileCoverage{
			Name:      profileName,
			TestCases: selected,
		})
		for _, testcase := range selected {
			selectedBy[testcase] = append(selectedBy[testcase], profileName)
		}
	}

	registeredSet := make(map[string]bool, len(registered))
	for _, name := range registered {
		registeredSet[name] = true
		profiles := selectedBy[name]
		if profiles == nil {
			profiles = []string{}
		}
		inventory.Testcases = append(inventory.Testcases, TestcaseCoverage{
			Name:     name,
			Profiles: profiles,
		})
	}

	for selected := range selectedBy {
		if !registeredSet[selected] {
			inventory.SelectedButUnregistered = append(inventory.SelectedButUnregistered, selected)
		}
	}
	sort.Strings(inventory.SelectedButUnregistered)
	if inventory.SelectedButUnregistered == nil {
		inventory.SelectedButUnregistered = []string{}
	}

	return inventory, nil
}

// Unreachable returns the registered testcases no profile selects, sorted.
func (inv Inventory) Unreachable() []string {
	unreachable := make([]string, 0)
	for _, testcase := range inv.Testcases {
		if len(testcase.Profiles) == 0 {
			unreachable = append(unreachable, testcase.Name)
		}
	}
	sort.Strings(unreachable)
	return unreachable
}

func registeredTestcaseNames() []string {
	testcases := pkgtestcases.List()
	names := make([]string, 0, len(testcases))
	for _, testcase := range testcases {
		names = append(names, testcase.Name)
	}
	return dedupSorted(names)
}

func dedupSorted(values []string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
