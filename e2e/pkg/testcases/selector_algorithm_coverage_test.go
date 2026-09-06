package testcases_test

import (
	"encoding/json"
	"os"
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
