// e2e/cmd/e2e-audit/main.go

// Command e2e-audit emits the runtime-derived E2E execution graph as
// deterministic JSON on stdout: registered testcases, canonical registered
// profiles, each profile's resolved GetTestCases selection, the derived
// drift sets, and the explicit per-profile CI selection overrides with
// their drift diagnosis. The output is generated audit evidence for issue
// #2379 and is never committed as a source of truth.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/vllm-project/semantic-router/e2e/pkg/verification"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// ciOverrideReport is the audit view of one explicit CI override: the list
// itself, the bounded exclusions recorded for it, and the three drift
// directions Gate D enforces (all empty on a consistent contract).
type ciOverrideReport struct {
	Profile            string   `json:"profile"`
	CITests            []string `json:"ci_tests"`
	RecordedExclusions []string `json:"recorded_exclusions"`
	OmittedUnrecorded  []string `json:"omitted_unrecorded"`
	StaleExclusions    []string `json:"stale_exclusions"`
	Phantom            []string `json:"phantom"`
}

func main() {
	inventory, err := verification.BuildInventory()
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: %v\n", err)
		os.Exit(1)
	}

	repoRoot, err := findRepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: %v\n", err)
		os.Exit(1)
	}
	workflowPath := filepath.Join(repoRoot, ".github", "workflows", "integration-test-k8s.yml")
	overrides, err := verification.LoadCIOverrides(workflowPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: %v\n", err)
		os.Exit(1)
	}

	selections := make(map[string][]string, len(inventory.Profiles))
	for _, profile := range inventory.Profiles {
		selections[profile.Name] = profile.TestCases
	}

	overrideReports := make([]ciOverrideReport, 0, len(overrides))
	for _, override := range overrides {
		exclusions := verification.KnownCIExclusions[override.Profile]
		recorded := make([]string, 0, len(exclusions))
		for name := range exclusions {
			recorded = append(recorded, name)
		}
		sort.Strings(recorded)

		drift := verification.DiagnoseCIOverrideDrift(selections[override.Profile], override.TestCases, exclusions)
		overrideReports = append(overrideReports, ciOverrideReport{
			Profile:            override.Profile,
			CITests:            override.TestCases,
			RecordedExclusions: recorded,
			OmittedUnrecorded:  drift.OmittedUnrecorded,
			StaleExclusions:    drift.StaleExclusions,
			Phantom:            drift.Phantom,
		})
	}

	output := struct {
		verification.Inventory
		Unreachable []string           `json:"registered_but_unreachable"`
		CIOverrides []ciOverrideReport `json:"ci_overrides"`
	}{
		Inventory:   inventory,
		Unreachable: inventory.Unreachable(),
		CIOverrides: overrideReports,
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: encoding inventory: %v\n", err)
		os.Exit(1)
	}
}

// findRepoRoot walks upward from the working directory to the enclosing git
// checkout, so the command resolves repository files regardless of which
// module directory it is invoked from.
func findRepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolving working directory: %w", err)
	}
	for {
		if _, statErr := os.Stat(filepath.Join(dir, ".git")); statErr == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("no git checkout found above the working directory; run e2e-audit from inside the repository")
		}
		dir = parent
	}
}
