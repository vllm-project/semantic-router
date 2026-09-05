// e2e/pkg/verification/gates_test.go

package verification

// The gates below enforce the three execution-graph invariants frozen for
// issue #2379 PR1. Registration side effects are provided here, at the test
// entrypoint, via the blank imports — the verification package itself owns
// no registration.

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"

	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

func buildInventoryForTest(t *testing.T) Inventory {
	t.Helper()
	inventory, err := BuildInventory()
	if err != nil {
		t.Fatalf("building runtime execution inventory: %v", err)
	}
	return inventory
}

// Gate A: registered-but-unreachable is an exact bounded debt ratchet.
//
// The invariant is unreachable == keys(KnownUnreachableDebt), checked in both
// directions: a newly unreachable testcase fails until it gets a profile
// mapping or a bounded disposition, and a resolved testcase fails until its
// stale debt entry is deleted. The debt table can only shrink intentionally.
func TestGateARegisteredButUnreachableMatchesKnownDebt(t *testing.T) {
	inventory := buildInventoryForTest(t)

	unreachable := make(map[string]bool)
	for _, name := range inventory.Unreachable() {
		unreachable[name] = true
	}

	unexpected := make([]string, 0)
	for name := range unreachable {
		if _, ok := KnownUnreachableDebt[name]; !ok {
			unexpected = append(unexpected, name)
		}
	}
	sort.Strings(unexpected)
	for _, name := range unexpected {
		t.Errorf("testcase %q is registered but unreachable from every registered profile; map it from a profile's GetTestCases() or record a bounded disposition in KnownUnreachableDebt", name)
	}

	stale := make([]string, 0)
	for name := range KnownUnreachableDebt {
		if !unreachable[name] {
			stale = append(stale, name)
		}
	}
	sort.Strings(stale)
	for _, name := range stale {
		t.Errorf("KnownUnreachableDebt entry %q is stale: the testcase is now reachable (or no longer registered); delete the resolved debt entry", name)
	}
}

// Every debt entry must carry exactly one bounded disposition: an owning
// child issue XOR a documented manual rationale.
func TestGateADebtEntriesAreBounded(t *testing.T) {
	for name, debt := range KnownUnreachableDebt {
		hasIssue := debt.Issue != 0
		hasRationale := debt.Rationale != ""
		if hasIssue == hasRationale {
			t.Errorf("KnownUnreachableDebt entry %q must set exactly one of Issue or Rationale", name)
		}
	}
}

// Gate B: selected-but-unregistered is empty, immediately and with no
// allowlist. This catches typos, stale testmatrix entries, and profile
// selections pointing at deleted registrations.
func TestGateBSelectedButUnregisteredIsEmpty(t *testing.T) {
	inventory := buildInventoryForTest(t)
	for _, name := range inventory.SelectedButUnregistered {
		profiles := make([]string, 0)
		for _, profile := range inventory.Profiles {
			for _, selected := range profile.TestCases {
				if selected == name {
					profiles = append(profiles, profile.Name)
				}
			}
		}
		t.Errorf("profile selection references unregistered testcase %q (selected by: %v); fix the name or register the testcase", name, profiles)
	}
}

// Gate C: runtime profile registration and the CI/profile registry agree
// exactly. This replaces source-regex reasoning about the Go execution side
// with actual runtime registration; the existing Python validation lane keeps
// its transitive registry <-> repo-manifest checks unchanged.
func TestGateCRuntimeProfilesMatchTestDomainRegistry(t *testing.T) {
	registryProfiles := loadTestDomainRegistryProfiles(t)

	runtimeProfiles := make(map[string]bool)
	for _, name := range framework.RegisteredProfileNames() {
		runtimeProfiles[name] = true
	}

	missingFromRegistry := make([]string, 0)
	for name := range runtimeProfiles {
		if !registryProfiles[name] {
			missingFromRegistry = append(missingFromRegistry, name)
		}
	}
	sort.Strings(missingFromRegistry)
	for _, name := range missingFromRegistry {
		t.Errorf("runtime-registered profile %q is missing from tools/agent/test-domain-registry.yaml", name)
	}

	missingFromRuntime := make([]string, 0)
	for name := range registryProfiles {
		if !runtimeProfiles[name] {
			missingFromRuntime = append(missingFromRuntime, name)
		}
	}
	sort.Strings(missingFromRuntime)
	for _, name := range missingFromRuntime {
		t.Errorf("tools/agent/test-domain-registry.yaml profile %q has no runtime registration in e2e/profiles/all", name)
	}
}

func loadTestDomainRegistryProfiles(t *testing.T) map[string]bool {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolving caller path for repo root")
	}
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..", "..")
	registryPath := filepath.Join(repoRoot, "tools", "agent", "test-domain-registry.yaml")

	raw, err := os.ReadFile(registryPath)
	if err != nil {
		t.Fatalf("reading %s: %v", registryPath, err)
	}

	var registry struct {
		Profiles map[string]any `yaml:"profiles"`
	}
	if err := yaml.Unmarshal(raw, &registry); err != nil {
		t.Fatalf("parsing %s: %v", registryPath, err)
	}
	if len(registry.Profiles) == 0 {
		t.Fatalf("%s declares no profiles; refusing to compare against an empty registry", registryPath)
	}

	profiles := make(map[string]bool, len(registry.Profiles))
	for name := range registry.Profiles {
		profiles[name] = true
	}
	return profiles
}
