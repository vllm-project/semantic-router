package framework

// These tests pin the effective-selection contract the runner executes.
// Expected sets are derived from the declared testmatrix data tables
// (BaselineRouterContract, BaselineStress) by independent set arithmetic in
// the test, never by calling the selector a second time, so a regression in
// EffectiveTestCases or testmatrix.BaselineCases cannot hide behind itself.

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
)

// contractMinusStress computes BaselineRouterContract \ BaselineStress by
// plain set difference, preserving contract order.
func contractMinusStress() []string {
	stress := make(map[string]bool, len(testmatrix.BaselineStress))
	for _, name := range testmatrix.BaselineStress {
		stress[name] = true
	}
	out := make([]string, 0, len(testmatrix.BaselineRouterContract))
	for _, name := range testmatrix.BaselineRouterContract {
		if !stress[name] {
			out = append(out, name)
		}
	}
	return out
}

func TestEffectiveTestCasesExplicitListBypassesProfileAndSuite(t *testing.T) {
	explicit := []string{"performance-throughput", "performance-latency"}
	opts := &TestOptions{Profile: BaselineSuiteProfile, BaselineSuite: "not-a-suite", TestCases: explicit}

	got, source, err := EffectiveTestCases(BaselineSuiteProfile, []string{"chat-completions-request"}, opts)
	if err != nil {
		t.Fatalf("explicit -tests must not consult the baseline suite: %v", err)
	}
	if source != SelectionExplicit {
		t.Fatalf("source = %q, want %q", source, SelectionExplicit)
	}
	if !reflect.DeepEqual(got, explicit) {
		t.Fatalf("effective = %v, want the explicit list %v", got, explicit)
	}
	got[0] = "mutated"
	if explicit[0] == "mutated" {
		t.Fatal("effective selection aliases the caller's -tests slice")
	}
}

func TestEffectiveTestCasesBaselineProfileStandardSuiteDropsExactlyStress(t *testing.T) {
	want := contractMinusStress()
	if len(want) == len(testmatrix.BaselineRouterContract) {
		t.Fatal("test precondition: BaselineStress must remove at least one contract case for this pin to mean anything")
	}

	for _, suite := range []string{DefaultBaselineSuite, ""} {
		opts := &TestOptions{Profile: BaselineSuiteProfile, BaselineSuite: suite}
		got, source, err := EffectiveTestCases(BaselineSuiteProfile, testmatrix.BaselineRouterContract, opts)
		if err != nil {
			t.Fatalf("suite %q: %v", suite, err)
		}
		if source != SelectionBaselineSuite {
			t.Fatalf("suite %q: source = %q, want %q", suite, source, SelectionBaselineSuite)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("suite %q: effective = %v, want BaselineRouterContract minus BaselineStress = %v", suite, got, want)
		}
	}
}

func TestEffectiveTestCasesBaselineProfileFullSuiteIsTheWholeContract(t *testing.T) {
	opts := &TestOptions{Profile: BaselineSuiteProfile, BaselineSuite: "full"}
	got, source, err := EffectiveTestCases(BaselineSuiteProfile, testmatrix.BaselineRouterContract, opts)
	if err != nil {
		t.Fatal(err)
	}
	if source != SelectionBaselineSuite {
		t.Fatalf("source = %q, want %q", source, SelectionBaselineSuite)
	}
	if !reflect.DeepEqual(got, testmatrix.BaselineRouterContract) {
		t.Fatalf("full suite = %v, want BaselineRouterContract verbatim", got)
	}
}

func TestEffectiveTestCasesBaselineProfileRejectsUnknownSuite(t *testing.T) {
	opts := &TestOptions{Profile: BaselineSuiteProfile, BaselineSuite: "stnadard"}
	if _, _, err := EffectiveTestCases(BaselineSuiteProfile, testmatrix.BaselineRouterContract, opts); err == nil {
		t.Fatal("unknown baseline suite accepted for the baseline profile")
	}
}

// Other profiles run their GetTestCases selection verbatim; the suite value
// is not consulted at all, so even an unknown suite is tolerated there. This
// pins current runner behavior on purpose: widening the suite layer to more
// profiles must change this test and the verification gates together.
func TestEffectiveTestCasesOtherProfilesIgnoreTheSuite(t *testing.T) {
	selection := []string{"chat-completions-request", "dashboard-health"}
	for _, suite := range []string{"", DefaultBaselineSuite, "full", "stnadard"} {
		opts := &TestOptions{Profile: "dashboard", BaselineSuite: suite}
		got, source, err := EffectiveTestCases("dashboard", selection, opts)
		if err != nil {
			t.Fatalf("suite %q: non-baseline profile must not fail on the suite: %v", suite, err)
		}
		if source != SelectionProfile {
			t.Fatalf("suite %q: source = %q, want %q", suite, source, SelectionProfile)
		}
		if !reflect.DeepEqual(got, selection) {
			t.Fatalf("suite %q: effective = %v, want the profile selection %v", suite, got, selection)
		}
	}
}
