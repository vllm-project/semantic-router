package testcases_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
)

type selectorCoverageEntry struct {
	Algorithm     string `json:"algorithm"`
	Tier          string `json:"tier"`
	Status        string `json:"status"`
	Profile       string `json:"profile"`
	TestCase      string `json:"testcase"`
	TargetProfile string `json:"target_profile"`
	Prerequisite  string `json:"prerequisite"`
}

func TestImplementedSelectorCoverageIsReachable(t *testing.T) {
	entries := loadSelectorCoverage(t)
	for _, entry := range entries {
		entry := entry
		t.Run(entry.Algorithm, func(t *testing.T) {
			switch entry.Status {
			case "planned":
				if entry.TargetProfile == "" {
					t.Fatal("planned coverage requires target_profile")
				}
				if entry.Profile != "" || entry.TestCase != "" {
					t.Fatal("planned coverage must not claim a runnable profile or testcase")
				}
			case "partial", "covered":
				assertCoverageIsReachable(t, entry)
			default:
				t.Fatalf("unknown coverage status %q", entry.Status)
			}
		})
	}
}

func assertCoverageIsReachable(t *testing.T, entry selectorCoverageEntry) {
	t.Helper()
	if entry.Profile == "" || entry.TestCase == "" {
		t.Fatal("implemented coverage requires profile and testcase")
	}
	if _, ok := pkgtestcases.Get(entry.TestCase); !ok {
		t.Fatalf("testcase %q is not registered", entry.TestCase)
	}
	profile, err := framework.NewProfileByName(entry.Profile)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range profile.GetTestCases() {
		if name == entry.TestCase {
			return
		}
	}
	t.Fatalf("testcase %q is unreachable from profile %q", entry.TestCase, entry.Profile)
}

func loadSelectorCoverage(t *testing.T) []selectorCoverageEntry {
	t.Helper()
	raw, err := os.ReadFile("testdata/selector_algorithm_coverage.json")
	if err != nil {
		t.Fatal(err)
	}
	var entries []selectorCoverageEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("selector coverage manifest is empty")
	}
	return entries
}

func TestSelectorAlgorithmCoverageTracksRuntimeCatalog(t *testing.T) {
	coverage := loadSelectorCoverage(t)
	byAlgorithm := make(map[string]selectorCoverageEntry, len(coverage))
	for _, entry := range coverage {
		if entry.Algorithm == "" {
			t.Fatal("selector coverage entry has empty algorithm")
		}
		if _, exists := byAlgorithm[entry.Algorithm]; exists {
			t.Errorf("selector coverage has duplicate entry for %q", entry.Algorithm)
		}
		byAlgorithm[entry.Algorithm] = entry
	}

	selectorCount := 0
	for _, algorithm := range loadRuntimeAlgorithmCatalog(t) {
		if algorithm.Execution != "selector" {
			continue
		}
		selectorCount++
		entry, ok := byAlgorithm[algorithm.Type]
		if !ok {
			t.Errorf("selector algorithm %q has no E2E coverage entry", algorithm.Type)
			continue
		}
		if entry.Tier != algorithm.Tier {
			t.Errorf("selector algorithm %q coverage tier = %q, runtime tier = %q", algorithm.Type, entry.Tier, algorithm.Tier)
		}
		delete(byAlgorithm, algorithm.Type)
	}

	if len(coverage) != selectorCount {
		t.Errorf("selector coverage has %d entries, runtime catalog has %d selectors", len(coverage), selectorCount)
	}
	for algorithm := range byAlgorithm {
		t.Errorf("selector coverage entry %q is absent from the runtime selector catalog", algorithm)
	}
}

// Run inside the router module so E2E can inspect the public catalog without
// adding the router's dependency graph to the E2E module.
func loadRuntimeAlgorithmCatalog(t *testing.T) []struct{ Type, Tier, Execution string } {
	t.Helper()
	helper, err := filepath.Abs("testdata/algorithm_catalog.go")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "run", helper)
	cmd.Dir = "../../../src/semantic-router"
	raw, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			t.Fatalf("read runtime algorithm catalog: %v\n%s", err, exitErr.Stderr)
		}
		t.Fatal(err)
	}
	var catalog []struct{ Type, Tier, Execution string }
	if err := json.Unmarshal(raw, &catalog); err != nil {
		t.Fatal(err)
	}
	return catalog
}
