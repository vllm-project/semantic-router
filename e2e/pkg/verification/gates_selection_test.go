package verification

// Gate E below extends the derived execution contract with the baseline-
// suite selection layer #3501 left outside the audit: the runner replaces
// the baseline profile's GetTestCases selection with testmatrix.BaselineCases
// (dispatched from CI via E2E_BASELINE_SUITE). Registration side effects for
// the live gates come from the blank imports in gates_test.go. The synthetic
// pin tests at the bottom exercise the pure drift diagnosis against fixed
// inputs, so each scenario stays pinned regardless of live registry state.

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
)

func deriveEffectiveSelectionForTest(t *testing.T) (Inventory, EffectiveSelection) {
	t.Helper()
	inventory := buildInventoryForTest(t)
	selection, err := DeriveEffectiveSelection(inventory)
	if err != nil {
		t.Fatalf("deriving effective selection: %v", err)
	}
	return inventory, selection
}

func profileSelectionForTest(t *testing.T, selection EffectiveSelection, profile string) ProfileSelection {
	t.Helper()
	for _, entry := range selection.Profiles {
		if entry.Profile == profile {
			return entry
		}
	}
	t.Fatalf("profile %q has no derived effective selection", profile)
	return ProfileSelection{}
}

// Gate E: the baseline suite layer narrows the baseline profile's runtime
// selection only by recorded exclusions.
//
// For the profile the runner narrows by suite, every GetTestCases case must
// execute under at least one accepted suite (no declared-but-never-run
// coverage), every per-suite omission must be explained by the production
// stress table, no suite may execute a case the profile does not select, and
// no stress entry may point at a case the profile does not select.
func TestGateEBaselineSuiteLayerMatchesProfileContract(t *testing.T) {
	inventory, selection := deriveEffectiveSelectionForTest(t)

	registered := false
	for _, profile := range inventory.Profiles {
		if profile.Name == selection.BaselineProfile {
			registered = true
		}
	}
	if !registered {
		t.Fatalf("framework.BaselineSuiteProfile names %q, which has no runtime registration; the suite layer narrows nothing", selection.BaselineProfile)
	}

	var selected []string
	for _, profile := range inventory.Profiles {
		if profile.Name == selection.BaselineProfile {
			selected = profile.TestCases
		}
	}
	entry := profileSelectionForTest(t, selection, selection.BaselineProfile)
	effectiveBySuite := make(map[string][]string, len(entry.Suites))
	for _, suite := range entry.Suites {
		if suite.Source != string(framework.SelectionBaselineSuite) {
			t.Errorf("baseline profile %q suite %q resolved via %q, want %q; the runner no longer applies the suite layer to it", selection.BaselineProfile, suite.Suite, suite.Source, framework.SelectionBaselineSuite)
		}
		effectiveBySuite[suite.Suite] = suite.TestCases
	}

	drift := DiagnoseSuiteDrift(selected, effectiveBySuite, testmatrix.BaselineStress)
	for _, name := range drift.OmittedFromAllSuites {
		t.Errorf("profile %q selects testcase %q but no baseline suite executes it; add it to testmatrix.BaselineRouterContract or drop it from the profile selection", selection.BaselineProfile, name)
	}
	for _, finding := range drift.UnrecordedExclusions {
		t.Errorf("baseline suite %q omits testcase %q that profile %q selects, and no production table records why; list it in testmatrix.BaselineStress or stop omitting it", finding.Suite, finding.Name, selection.BaselineProfile)
	}
	for _, finding := range drift.Phantom {
		t.Errorf("baseline suite %q executes testcase %q which profile %q does not select at runtime; fix the suite contract or the profile selection", finding.Suite, finding.Name, selection.BaselineProfile)
	}
	for _, name := range drift.StaleStress {
		t.Errorf("testmatrix.BaselineStress entry %q is not selected by profile %q; delete the stale entry", name, selection.BaselineProfile)
	}
}

// Gate E: every profile other than the baseline profile executes its
// GetTestCases selection unchanged under every accepted suite. A runner
// change that widens the suite layer surfaces here before the audit view
// silently under-reports what those profiles run.
func TestGateESuiteLayerLeavesOtherProfilesUntouched(t *testing.T) {
	inventory, selection := deriveEffectiveSelectionForTest(t)
	for _, profile := range inventory.Profiles {
		if profile.Name == selection.BaselineProfile {
			continue
		}
		entry := profileSelectionForTest(t, selection, profile.Name)
		for _, suite := range entry.Suites {
			if suite.Source != string(framework.SelectionProfile) {
				t.Errorf("profile %q suite %q resolved via %q, want %q", profile.Name, suite.Suite, suite.Source, framework.SelectionProfile)
			}
			if !reflect.DeepEqual(suite.TestCases, profile.TestCases) {
				t.Errorf("profile %q suite %q executes %v, want its GetTestCases selection %v", profile.Name, suite.Suite, suite.TestCases, profile.TestCases)
			}
			if len(suite.Excluded) != 0 {
				t.Errorf("profile %q suite %q records exclusions %v; only the baseline profile is narrowed by suite", profile.Name, suite.Suite, suite.Excluded)
			}
		}
	}
}

// Gate E: the CI workflow dispatches a suite the runtime accepts, through
// the environment variable the e2e binary reads. The workflow input default
// is the suite every non-full CI plan runs, so it must be one
// testmatrix.BaselineCases resolves rather than a value the runner rejects
// after the profile is already deployed.
func TestGateECIWorkflowDispatchesAcceptedSuite(t *testing.T) {
	workflowPath := filepath.Join(repoRootForTest(t), ".github", "workflows", "integration-test-k8s.yml")
	dispatch, err := LoadCISuiteDispatch(workflowPath)
	if err != nil {
		t.Fatalf("loading CI suite dispatch: %v", err)
	}

	if dispatch.EnvName != framework.BaselineSuiteEnv {
		t.Errorf("CI workflow carries the baseline suite in %s but the e2e binary reads %s", dispatch.EnvName, framework.BaselineSuiteEnv)
	}
	if _, err := testmatrix.BaselineCases(dispatch.InputDefault); err != nil {
		t.Errorf("CI workflow input default %q is rejected by testmatrix.BaselineCases: %v", dispatch.InputDefault, err)
	}
	if dispatch.InputDefault != framework.DefaultBaselineSuite {
		t.Errorf("CI workflow input default %q differs from the runner default %q; CI and local runs would execute different baseline scopes by default", dispatch.InputDefault, framework.DefaultBaselineSuite)
	}
}

// Synthetic pins for the pure drift diagnosis. Each scenario uses fixed
// inputs so it is caught even when the live contract is clean.

func TestSuiteDriftPinsOmittedFromAllSuites(t *testing.T) {
	selected := []string{"a", "b", "never-run"}
	effective := map[string][]string{
		"standard": {"a", "b"},
		"full":     {"a", "b"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, nil)
	if !reflect.DeepEqual(drift.OmittedFromAllSuites, []string{"never-run"}) {
		t.Fatalf("OmittedFromAllSuites = %v, want [never-run]", drift.OmittedFromAllSuites)
	}
	// The same case is also an unrecorded exclusion under each suite: the
	// two findings describe one defect from two angles and both must fire.
	want := []SuiteFinding{{Suite: "full", Name: "never-run"}, {Suite: "standard", Name: "never-run"}}
	if !reflect.DeepEqual(drift.UnrecordedExclusions, want) {
		t.Fatalf("UnrecordedExclusions = %v, want %v", drift.UnrecordedExclusions, want)
	}
}

func TestSuiteDriftPinsRecordedStressExclusion(t *testing.T) {
	selected := []string{"a", "stress-1"}
	effective := map[string][]string{
		"standard": {"a"},
		"full":     {"a", "stress-1"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, []string{"stress-1"})
	if len(drift.OmittedFromAllSuites)+len(drift.UnrecordedExclusions)+len(drift.Phantom)+len(drift.StaleStress) != 0 {
		t.Fatalf("a stress case omitted only by the standard suite is legitimate, got %+v", drift)
	}
}

func TestSuiteDriftPinsUnrecordedExclusion(t *testing.T) {
	selected := []string{"a", "quietly-dropped"}
	effective := map[string][]string{
		"standard": {"a"},
		"full":     {"a", "quietly-dropped"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, nil)
	want := []SuiteFinding{{Suite: "standard", Name: "quietly-dropped"}}
	if !reflect.DeepEqual(drift.UnrecordedExclusions, want) {
		t.Fatalf("UnrecordedExclusions = %v, want %v", drift.UnrecordedExclusions, want)
	}
	if len(drift.OmittedFromAllSuites) != 0 {
		t.Fatalf("a case the full suite runs is not omitted from all suites, got %v", drift.OmittedFromAllSuites)
	}
}

func TestSuiteDriftPinsPhantom(t *testing.T) {
	selected := []string{"a"}
	effective := map[string][]string{
		"standard": {"a", "ghost"},
		"full":     {"a"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, nil)
	want := []SuiteFinding{{Suite: "standard", Name: "ghost"}}
	if !reflect.DeepEqual(drift.Phantom, want) {
		t.Fatalf("Phantom = %v, want %v", drift.Phantom, want)
	}
}

func TestSuiteDriftPinsStaleStress(t *testing.T) {
	selected := []string{"a"}
	effective := map[string][]string{
		"standard": {"a"},
		"full":     {"a"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, []string{"retired-stress"})
	if !reflect.DeepEqual(drift.StaleStress, []string{"retired-stress"}) {
		t.Fatalf("StaleStress = %v, want [retired-stress]", drift.StaleStress)
	}
}

func TestSuiteDriftCleanContract(t *testing.T) {
	selected := []string{"a", "b", "stress-1"}
	effective := map[string][]string{
		"standard": {"a", "b"},
		"full":     {"a", "b", "stress-1"},
	}
	drift := DiagnoseSuiteDrift(selected, effective, []string{"stress-1"})
	want := SuiteDrift{
		OmittedFromAllSuites: []string{},
		UnrecordedExclusions: []SuiteFinding{},
		Phantom:              []SuiteFinding{},
		StaleStress:          []string{},
	}
	if !reflect.DeepEqual(drift, want) {
		t.Fatalf("clean contract reported drift: %+v", drift)
	}
}
